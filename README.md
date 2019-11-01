# Deformable Text Recognition
This software implements the Deformable Convolutional Recurrent Neural Network, a combination of of Convolutional Recurrent Neural Network, Deformable Convolutional Networks and Residual Blocks. Some of the codes are from [crnn.pytorch](https://github.com/meijieru/crnn.pytorch) and [Deformable-ConvNets](https://github.com/msracver/Deformable-ConvNets). For details, please refer to our paper https://arxiv.org/abs/1908.10998.

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

Each line should end with '\n'. You can also use the data provided by us, which is contained in the test_data folder.

## Train and Test
To train a new model, simply execute python train.py --lmdb_paths {train_path_list} --cuda. If you need to set other parameters, explore train.py for details.

To test a trained model, you need to explore and execute eval.py.

## Citation
    @misc{deng2019focusenhanced,
          title={Focus-Enhanced Scene Text Recognition with Deformable Convolutions},
          author={Linjie Deng and Yanxiang Gong and Xinchen Lu and Xin Yi and Zheng Ma and Mei Xie},
          year={2019},
          eprint={1908.10998},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
    }
