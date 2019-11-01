import torch.nn as nn
from torch_deform_conv.layers import ConvOffset2D


class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


class DeformableCRNN(nn.Module):
    def __init__(self, imgH, nc, nclass, nh=256):
        super(DeformableCRNN, self).__init__()
        assert imgH % 16 == 0, 'imgH has to be a multiple of 16'

        self.conv0 = nn.Conv2d(nc, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pooling0 = nn.AdaptiveMaxPool2d(output_size=(32, 100))
        self.conv1 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pooling1 = nn.AdaptiveMaxPool2d(output_size=(16, 50))
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.deconv3 = ConvOffset2D(256, 1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pooling2 = nn.AdaptiveMaxPool2d(output_size=(8, 25))

        self.resblock1 = ResidualBlock(256)
        self.resblock2 = ResidualBlock(256)
        self.resblock3 = ResidualBlock(256)
        self.resblock4 = ResidualBlock(256)

        self.deconv4 = ConvOffset2D(256, 1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pooling3 = nn.AdaptiveMaxPool2d(output_size=(4, 26))
        self.conv6 = nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.pooling4 = nn.AdaptiveMaxPool2d(output_size=(2, 27))

        self.conv7 = nn.Conv2d(512, 512, kernel_size=(2, 2), stride=(1, 1))
        self.bn6 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True)

        self.relu = nn.ReLU(True)
        self.p_relu = nn.PReLU()

        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass))

    def forward(self, input_):
        e1 = self.pooling0(self.relu(self.conv0(input_)))
        e2 = self.pooling1(self.relu(self.conv1(e1)))
        e3 = self.relu(self.bn2(self.conv2(e2)))
        e4_ = self.pooling2(self.relu(self.conv3(self.deconv3(e3))))

        e4 = self.resblock4(self.resblock3(self.resblock2(self.resblock1(e4_))))

        e5 = self.relu(self.bn4(self.conv4(self.deconv4(e4))))
        e6 = self.pooling3(self.relu(self.conv5(e5)))
        e7 = self.pooling4(self.relu(self.conv6(e6)))
        e8 = self.relu(self.bn6(self.conv7(e7)))

        b, c, h, w = e8.size()
        assert h == 1, "the height of conv must be 1"
        e8 = e8.squeeze(2)
        e8 = e8.permute(2, 0, 1)  # [w, b, c]
        # rnn features
        output = self.rnn(e8)

        return output


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.rconv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.rconv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.rconv1(x)
        residual = self.bn1(residual)
        residual = self.relu(residual)
        residual = self.rconv2(residual)
        residual = self.bn2(residual)

        return x + residual
