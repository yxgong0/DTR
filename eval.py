from models import deformable_crnn
import torch
import utils
import os
from torch.autograd import Variable
from dataset import TestDataset

# Testing parameters
dataset_names = ['ICDAR03', 'ICDAR13', 'ICDAR15', 'Total', 'SVT', 'IIIT5K', 'SVTP']  # the folder names
_batch_size = 64
colored = False
case_sensitive = False
img_w, img_h = 100, 32
workers = 8
use_cuda = True
generate_txt_file = False  # whether to generate the file of the testing result

nc = 3 if colored else 1
alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ' \
    if case_sensitive else '0123456789abcdefghijklmnopqrstuvwxyz'
nclass = len(alphabet) + 1

test_path = './test_data/'
model_path = 'deformable_crnn.pth'

image = torch.FloatTensor(_batch_size, 3, img_w, img_h)
text = torch.IntTensor(_batch_size * 5)
length = torch.IntTensor(_batch_size)
converter = utils.strLabelConverter(alphabet)

if use_cuda:
    image = image.cuda()

if generate_txt_file:
    if not os.path.exists('./test_results/'):
        os.system('mkdir {0}'.format('./test_results/'))


def val(network, dataset, dataset_name):
    print('Starting val of ' + dataset_name + ' ...')

    for p in network.parameters():
        p.requires_grad = False

    network.eval()
    data_loader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=_batch_size, num_workers=int(workers))
    val_iter = iter(data_loader)

    n_correct = 0

    max_iter = len(data_loader)

    if generate_txt_file:
        result_txt_file = open('./test_results/' + dataset_name + '_results.txt', 'w+')

    for i in range(max_iter):
        data = val_iter.next()
        i += 1
        cpu_images, cpu_texts, image_name = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, le = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, le)

        preds = network(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target, im_name in zip(sim_preds, cpu_texts, image_name):
            if generate_txt_file:
                result_txt_file.write(im_name + ', "' + pred + '"\r\n')
            if pred == target.lower():
                n_correct += 1

    if generate_txt_file:
        result_txt_file.close()

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * _batch_size)
    return accuracy


if __name__ == '__main__':
    net_ = deformable_crnn.DeformableCRNN(img_h, nc, nclass, 256)
    if use_cuda:
        net_ = net_.cuda()
        net = torch.nn.DataParallel(net_, device_ids=range(1))
    para = torch.load(model_path)
    net_.load_state_dict(para)

    results = []
    for dataset_name_ in dataset_names:
        test_dataset = TestDataset(list_file=test_path + dataset_name_ + '/gt.txt',
                                   alphabet=alphabet, transform=utils.ResizeNormalize((img_w, img_h)),
                                   colored=colored)
        accuracy_ = val(net_, test_dataset, dataset_name_)
        results.append((dataset_name_, accuracy_))
    for result in results:
        print('Accuracy of ', result[0], ' is ', result[1])
