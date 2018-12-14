import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16


class simpleNet(nn.Module):
    def __init__(self, unit):
        super(simpleNet, self).__init__()
        self.fc1 = nn.Linear(28*28, unit)
        self.fc2 = nn.Linear(unit, unit)
        self.fc3 = nn.Linear(unit, 10)

    def forward(self, x):
        ## flatten
        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)

        return x

    def num_flat_features(self, x):
        # Conv2dは入力を4階のテンソルとして保持する(サンプル数*チャネル数*縦の長さ*横の長さ)
        # よって、特徴量の数を数える時は[1:]でスライスしたものを用いる
        size = x.size()[1:] ## all dimensions except the batch dimension
        # 特徴量の数=チャネル数*縦の長さ*横の長さを計算する
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class simpleCNN(nn.Module):

    def __init__(self):
        super(simpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(5, 10, kernel_size=5, stride=2, padding=2)
        self.l_out = nn.Linear(490, 10)

    def forward(self, x):
        # 入力→畳み込み層1→活性化関数(ReLU)→プーリング層1(2*2)→出力
        # input 28 x 28 x 1
        # conv1 28 x 28 x 1 -> 24 x 24 x 10
        # max_pool(kernel2) 12 x 12 x 10
        x = self.conv1(x)
        x = F.relu(x)
        # x = F.max_pool2d(x, (2 ,2) )

        # 入力→畳み込み層2→活性化関数(ReLU)→プーリング層2(2*2)→出力
        # conv2 12 x 12 x 10 -> 8 x 8 x 20
        # max_pool(kernel2) -> 4 x 4 x 20
        x = self.conv2(x)
        x = F.relu(x)
        # x = F.max_pool2d(x, 2)

        # x = self.conv2_drop(x)
        # output layer
        # x = x.view(-1, self.num_flat_features(x))
        # self.num_flat_featuresで特徴量の数を算出
        # flatten 4 x 4 x 20 = 320
        x = x.view(-1, self.num_flat_features(x))
        x = self.l_out(x)
        x = F.log_softmax(x, dim=1)

        return x


    def num_flat_features(self, x):
        # Conv2dは入力を4階のテンソルとして保持する(サンプル数*チャネル数*縦の長さ*横の長さ)
        # よって、特徴量の数を数える時は[1:]でスライスしたものを用いる
        size = x.size()[1:]  ## all dimensions except the batch dimension
        # 特徴量の数=チャネル数*縦の長さ*横の長さを計算する
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 畳み込み層を定義する
        # 引数は順番に、サンプル数、チャネル数、フィルタのサイズ

        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, padding=0)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=1, padding=0)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=0)
        self.conv6 = nn.Conv2d(128, 128, kernel_size=5, padding=0)

        self.conv7 = nn.Conv2d(128, 256, kernel_size=1, padding=0)
        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, padding=0)
        self.conv9 = nn.Conv2d(256, 256, kernel_size=5, padding=0)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(128)
        self.bn6 = nn.BatchNorm2d(128)
        self.bn7 = nn.BatchNorm2d(256)
        self.bn8 = nn.BatchNorm2d(256)
        self.bn9 = nn.BatchNorm2d(256)

        self.drop = nn.Dropout2d(p=0.2)
        self.fc1 = nn.Linear(256, 10)
        # self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.drop(x)
        x = F.max_pool2d(x, (2, 2))

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.relu(x)
        x = self.drop(x)

        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)
        x = self.conv9(x)
        x = self.bn9(x)
        x = F.relu(x)
        x = self.drop(x)

        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = F.log_softmax(x, dim=1)

        return x

    def num_flat_features(self, x):
        # Conv2dは入力を4階のテンソルとして保持する(サンプル数*チャネル数*縦の長さ*横の長さ)
        # よって、特徴量の数を数える時は[1:]でスライスしたものを用いる
        size = x.size()[1:]  ## all dimensions except the batch dimension
        # 特徴量の数=チャネル数*縦の長さ*横の長さを計算する
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



class CNNold(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 畳み込み層を定義する
        # 引数は順番に、サンプル数、チャネル数、フィルタのサイズ

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv6= nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.drop = nn.Dropout2d(p=0.2)
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)

        self.fc1 = nn.Linear(1024, 256)
        self.fc2 = nn.Linear(256, 10)
        # self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2 ,2) )
        # x = self.drop(x)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.drop(x)

        x = self.conv5(x)
        x = self.conv6(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # x = self.drop(x)

        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x

    def num_flat_features(self, x):
        # Conv2dは入力を4階のテンソルとして保持する(サンプル数*チャネル数*縦の長さ*横の長さ)
        # よって、特徴量の数を数える時は[1:]でスライスしたものを用いる
        size = x.size()[1:] ## all dimensions except the batch dimension
        # 特徴量の数=チャネル数*縦の長さ*横の長さを計算する
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class VGG_New(nn.Module):

    def __init__(self, features, classifiers, device, num_classes=10, init_weights=True, feature_grad=0, classifier_grad=0):
        super(VGG_New, self).__init__()
        # self.features = features

        ## CNN層は学習しない

        print(' learn feature layers upto {} layers'.format(feature_grad))
        print(' learn classifier layers upto {} layers'.format(classifier_grad))

        for f in features[:feature_grad]:
            for p in f.parameters():
                p.requires_grad = False

        for f in classifiers[:classifier_grad]:
            for p in f.parameters():
                p.requires_grad = False

        classifiers[6] = nn.Linear(4096, num_classes)

        self.features = features.to(device)
        self.classifier = classifiers.to(device)

        #
        # # change here with you code
        # self.classifier = nn.Sequential(
        #     nn.Linear(512 * 7 * 7, 512),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(512, 512),
        #     nn.ReLU(True),
        #     nn.Dropout(),
        #     nn.Linear(512, num_classes),
        # )

        # if init_weights:
        #     print(' set initial weight')
        #     self._initialize_weights(self.features)
        #
        # self._initialize_weights(self.classifier)
        # self.features = nn.DataParallel(net.features, device_ids=[0, 1, 2, 3]).cuda()
        # self.classifier = nn.DataParallel(net.classifier, device_ids=[0, 1, 2, 3]).cuda()

        # if init_weights:
        #     self._initialize_weights()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # x = F.log_softmax(x, dim=1)

        return x

    def _initialize_weights(self, models):
        for m in models:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

class myVGG16(nn.Module):

    def __init__(self):
        super(myVGG16, self).__init__()
        # 畳み込み層を定義する
        # 引数は順番に、サンプル数、チャネル数、フィルタのサイズ
        self.conv_block1 = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block2 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block3 = nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block4 = nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv_block5 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2)
        )


        self.fc3 = nn.Linear(512, 10)

        # self.fc3 = nn.Linear(256, 10)

    def forward(self, x):


        h = self.conv_block1(x)
        h = self.conv_block2(h)
        h = self.conv_block3(h)
        h = self.conv_block4(h)
        h = self.conv_block5(h)
        h = h.view(-1, self.num_flat_features(h))
        h = self.fc3(h)

        return h

    def num_flat_features(self, x):
        # Conv2dは入力を4階のテンソルとして保持する(サンプル数*チャネル数*縦の長さ*横の長さ)
        # よって、特徴量の数を数える時は[1:]でスライスしたものを用いる
        size = x.size()[1:] ## all dimensions except the batch dimension
        # 特徴量の数=チャネル数*縦の長さ*横の長さを計算する
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}