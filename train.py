import argparse
import random
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import os
import utils
import models.deformable_crnn as crnn
from warpctc_pytorch import CTCLoss
import time
import torch.nn
from dataset import TestDataset
from dataset import LMDBDataset

parser = argparse.ArgumentParser()
parser.add_argument('--lmdb_paths', type=list,
                    default=['text_recognition_train/MJSYNTH_LMDB/', 'text_recognition_train/SYNTH_LMDB/'],
                    help='The list of paths to training data with LMDB format.')
parser.add_argument('--val_list', type=str, default='test_data/ICDAR13/gt.txt',
                    help='The list file of testing data names and annotations.')
parser.add_argument('--workers', type=int, default=8, help='The number of data loading workers.')
parser.add_argument('--batch_size', type=int, default=64, help='The input batch size.')
parser.add_argument('--img_h', type=int, default=64, help='The height of the input image to network.')
parser.add_argument('--img_w', type=int, default=200, help='The width of the input image to network.')
parser.add_argument('--colored', default=False, help='Whether to input colored images.')
parser.add_argument('--epochs', type=int, default=10, help='The number of epochs to train for.')
parser.add_argument('--lr', type=float, default=0.00005, help='The learning rate of the optimizer.')
parser.add_argument('--crnn', default='', help="The path to pre-trained crnn model.")
parser.add_argument('--output_path', default='./results/', help='Where to store samples and models.')
parser.add_argument('--display_interval', type=int, default=10, help='Interval to display the training information.')
parser.add_argument('--test_display_number', type=int, default=10, help='How many samples to display when testing.')
parser.add_argument('--val_interval', type=int, default=5000, help='The interval to validate the model.')
parser.add_argument('--save_interval', type=int, default=5000, help='The interval to save the model.')
parser.add_argument('--save_logfile', default=False, help='Whether to save the log file.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD.')
parser.add_argument('--cuda', action='store_true', help='Whether to enable cuda.')
parser.add_argument('--manual_seed', type=int, default=None,
                    help='A fix seed, which will be randomly generated if set to None.')
parser.add_argument('--case_sensitive', default=False, help='Whether the model is case sensitive.')
parser.add_argument('--alphabet_insensitive', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz',
                    help='The alphabet when using case insensitive mode.')
parser.add_argument('--alphabet sensitive', type=str,
                    default='0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ',
                    help='The alphabet when using case sensitive mode.')
opt = parser.parse_args()

if not os.path.exists(opt.output_path):
    os.system('mkdir {0}'.format(opt.output_path))

# Choose the alphabet
alphabet = opt.alphabet_sensitive if opt.case_sensitive else opt.alphabet_insensitive

# Set the seed
if opt.manual_seed is None:
    opt.manual_seed = random.randint(1, 10000)
random.seed(opt.manual_seed)
np.random.seed(opt.manual_seed)
torch.manual_seed(opt.manual_seed)

# Use cudnn
cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# Prepare training and testing data
train_datasets = [LMDBDataset(x) for x in opt.lmdb_paths]
train_dataset = torch.utils.data.ConcatDataset(train_datasets)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, sampler=None,
                                           num_workers=int(opt.workers),
                                           collate_fn=utils.AlignCollate(im_h=opt.img_h, im_w=opt.img_w))
print('Num of Training Images: %s' % len(train_dataset))

test_dataset = TestDataset(list_file=opt.val_list, alphabet=alphabet,
                           transform=utils.ResizeNormalize((opt.img_w, opt.img_h)))
print('Num of Testing Images: %s' % len(test_dataset))

nclass = len(alphabet) + 1
nc = 3 if opt.colored else 1

converter = utils.strLabelConverter(alphabet)
criterion = CTCLoss()


# custom weights initialization called on crnn
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


crnn = crnn.DeformableCRNN(opt.img_h, nc, nclass)
crnn.apply(weights_init)

# Load pre-trained model if provided
if opt.crnn != '':
    print('Loading pretrained model from %s ...' % opt.crnn)
    if opt.cuda:
        crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.gpu_number))
    crnn.load_state_dict(torch.load(opt.crnn))

image = torch.FloatTensor(opt.batch_size, 3, opt.img_h, opt.img_h)
text = torch.IntTensor(opt.batch_size * 5)
length = torch.IntTensor(opt.batch_size)

if opt.cuda:
    crnn.cuda()
    if crnn == '':
        crnn = torch.nn.DataParallel(crnn, device_ids=range(opt.gpu_number))
    image = image.cuda()
    criterion = criterion.cuda()

# loss averager
loss_avg = utils.averager()

# to use SGD optimizer
optimizer = optim.SGD(crnn.parameters(), lr=opt.lr, momentum=opt.momentum)


# Validate the model during the training process
def val(net, dataset, criterion, best_accuracy, epoch, i, best_epoch, best_i, max_iter=100):
    print('Validating...')

    for para in crnn.parameters():
        para.requires_grad = False

    net.eval()
    data_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=opt.batch_size, num_workers=int(opt.workers))
    val_iter = iter(data_loader)

    n_correct = 0
    loss_avg_ = utils.averager()

    max_iter = min(max_iter, len(data_loader))
    for j in range(max_iter):
        data = val_iter.next()
        j += 1
        cpu_images, cpu_texts, _ = data
        batch_size = cpu_images.size(0)
        utils.loadData(image, cpu_images)
        t, length_ = converter.encode(cpu_texts)
        utils.loadData(text, t)
        utils.loadData(length, length_)

        preds = crnn(image)
        preds_size = Variable(torch.IntTensor([preds.size(0)] * batch_size))
        cost_ = criterion(preds, text, preds_size, length) / batch_size
        loss_avg_.add(cost_)

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, cpu_texts):
            if pred == target.lower():
                n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:opt.test_display_number]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, cpu_texts):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = n_correct / float(max_iter * opt.batch_size)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_epoch = epoch
        best_i = i
    print('Best accuracy: ', best_accuracy, ' from ecpoch ', best_epoch, ', iteration ', best_i)
    return best_accuracy, best_epoch, best_i


def train_batch(model, criterion, optimizer):
    data = train_iter.next()
    cpu_images, cpu_texts = data
    batch_size = cpu_images.size(0)
    utils.loadData(image, cpu_images)
    t, length_ = converter.encode(cpu_texts)
    utils.loadData(text, t)
    utils.loadData(length, length_)

    preds = model(image)
    preds_size = torch.IntTensor([preds.size(0)] * batch_size)
    loss = criterion(preds, text, preds_size, length) / batch_size
    crnn.zero_grad()
    loss.backward()
    optimizer.step()
    return loss


if opt.save_logfile:
    log = open(opt.output_path + 'log.txt', 'w+')

start = time.time()
best_accuracy = 0
best_epoch = 0
best_i = 0
for epoch in range(opt.epochs):
    train_iter = iter(train_loader)
    i = 0
    time_train_batch = 0

    while i < len(train_loader):
        for p in crnn.parameters():
            p.requires_grad = True
        crnn.train()

        t1 = time.time()
        cost = train_batch(crnn, criterion, optimizer)
        t2 = time.time()
        loss_avg.add(cost)
        i += 1

        time_train_batch += (t2 - t1)

        if i % opt.display_interval == 0:
            time_train_batch = time_train_batch / opt.display_interval
            print('[%d/%d][%d/%d] Loss: %f Time: %.2f ms per batch' % (epoch, opt.epochs, i, len(train_loader),
                                                                       loss_avg.val(), time_train_batch * 1000))
            time_train_batch = 0
            if opt.save_logfile:
                log.write('[%d/%d][%d/%d] Loss: %f\n' % (epoch, opt.epochs, i, len(train_loader), loss_avg.val()))
            loss_avg.reset()

        if i % opt.val_interval == 0:
            best_accuracy, best_epoch, best_i = val(crnn, test_dataset, criterion, best_accuracy,
                                                    epoch, i, best_epoch, best_i)

        if i % opt.save_interval == 0:
            torch.save(crnn.state_dict(), '{0}/crnn_deform_{1}_{2}.pth'.format(opt.output_path, epoch, i))

end = time.time()
torch.save(crnn.state_dict(), 'crnn_deform_final.pth')
print('Program processed ', end - start, 's, ', (end - start)/60, 'min, ', (end - start)/3600, 'h')

if opt.save_logfile:
    log.write('Program processed ' + str(end - start) + 's, ' + str((end - start)/60) + 'min, ' +
              str((end - start)/3600) + 'h')
    log.close()
