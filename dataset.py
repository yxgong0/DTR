import os
from PIL import Image
from torch.utils.data import Dataset
import lmdb
import six
import sys


class TestDataset(Dataset):

    def __init__(self, list_file=None, alphabet=None, case_senstive=False,
                 colored=False, transform=None, target_transform=None):
        self._list_file = list_file
        self._root_dir = self._list_file[0:self._list_file.rindex('/')]
        self._alphabet = alphabet
        self._case_senstive = case_senstive
        self._colored = colored
        self._transform = transform
        self._target_transform = target_transform

        assert os.path.exists(list_file), 'List file does not exist: {}'.format(list_file)

        self._ims_list = self._load_image_list()

    def __len__(self):
        return len(self._ims_list)

    def __getitem__(self, index):
        im_name, label = self._ims_list[index]
        im_path = os.path.join(self._root_dir, im_name)
        if not self._colored:
            im = Image.open(im_path).convert('L')
        else:
            im = Image.open(im_path).convert('RGB')

        if self._transform is not None:
            im = self._transform(im)
        if self._target_transform is not None:
            label = self._target_transform(label)
        data = (im, label, im_name)
        return data

    def _load_image_list(self):
        assert os.path.exists(self._list_file), 'Path does not exist: {}'.format(self._list_file)
        ims_list = []
        with open(self._list_file) as f:
            n = 0
            for line in f.readlines():
                n += 1
                if not len(line.strip()):
                    continue
                name, label = self._extra_items(line)
                label = self._label_filter(label)
                if not label:
                    continue
                ims_list.append([name, label])

        return ims_list

    def _extra_items(self, str_):
        items = str_.split(' ')
        im_name, label = items[0], items[1]
        im_name = im_name if im_name[-1].isalnum() else im_name[:-1]
        label = label[1:-2]

        label = self._label_filter(label)
        return im_name, label

    def _label_filter(self, label):
        assert isinstance(label, str), 'Label is not a String.'
        if not self._case_senstive:
            label = label.lower()
        label = ''.join([x for x in label if x in self._alphabet])
        return label


class LMDBDataset(Dataset):

    def __init__(self, root=None, transform=None, target_transform=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)

        if not self.env:
            print('cannot creat lmdb from %s' % root)
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            samples = int(txn.get('num-samples'.encode()))
            self.nSamples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('L')
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            if self.transform is not None:
                img = self.transform(img)

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()))

            label = label[2:-1]
            label_new = ''
            for c in label:
                if c.isalnum():
                    label_new += c

            label = label_new

            if self.target_transform is not None:
                label = self.target_transform(label)

        data = (img, label)
        return data
