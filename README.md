# Deformable Text Recognition
This software implements the Deformable Convolutional Recurrent Neural Network, a combination of of Convolutional Recurrent Neural Network, Deformable Convolutional Networks and Residual Blocks. Some of the codes are from [crnn.pytorch](https://github.com/meijieru/crnn.pytorch) and [Deformable-ConvNets](https://github.com/msracver/Deformable-ConvNets). For details, please refer to [our paper](https://ieeexplore.ieee.org/abstract/document/9064428).

## Requirements
* [Python 3.6](https://www.python.org/)
* [PyTorch 1.0](https://pytorch.org/)
* [TorchVision](https://pypi.org/project/torchvision/)
* [Numpy](https://pypi.org/project/numpy/)
* [Six](https://pypi.org/project/six/)
* [Scipy](https://pypi.org/project/scipy/)
* [LMDB](https://pypi.org/project/lmdb/)
* [Pillow](https://pypi.org/project/Pillow/) 
* [warp-ctc-pytorch](https://github.com/baidu-research/warp-ctc)

## Data Preparation
Please convert your own training dataset to LMDB format. The testing images should be in one folder with a txt file, and the txt file contains some lines which contains the filename and the label, formatted as:

img_001.png, "word"

Each line should end with '\n'. You can also use the data provided by us from 
* [Baidu Netdisk](https://pan.baidu.com/s/10NHaiJaRO1TpMON-OgcPFQ)
* [Google Drive](https://drive.google.com/open?id=1z48dRxyFVCjokXYRQtFx6k3bGgh52EzL)
* [Onedrive](https://1drv.ms/u/s!Aoxs3QdtBgqEssZAw5YBe_iOYPiyuw?e=SiXnrE)

which should be contained in the test_data folder.

## Train and Test
To train a new model, simply execute python train.py --lmdb_paths {train_path_list} --cuda. If you need to set other parameters, explore train.py for details.

To test a trained model, you need to explore and execute eval.py.

## Citation
    @inproceedings{deng2019focus,
      title={Focus-Enhanced Scene Text Recognition with Deformable Convolutions},
      author={Deng, Linjie and Gong, Yanxiang and Lu, Xinchen and Yi, Xin and Ma, Zheng and Xie, Mei},
      booktitle={2019 IEEE 5th International Conference on Computer and Communications (ICCC)},
      pages={1685--1689},
      year={2019},
      organization={IEEE}
    }
