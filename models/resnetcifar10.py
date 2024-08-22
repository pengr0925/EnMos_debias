import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

_BATCH_NORM_DECAY = 0.1
_BATCH_NORM_EPSILON = 1e-5


def batch_norm2d(num_features):
    return nn.BatchNorm2d(num_features, eps=_BATCH_NORM_EPSILON, momentum=_BATCH_NORM_DECAY)


def _weights_init(layer):
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        init.kaiming_normal_(layer.weight)
    # elif isinstance(layer, nn.BatchNorm2d):
    #     layer.weight.data.fill_(1)
    #     layer.bias.data.zero_()


class IdentityBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = batch_norm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = batch_norm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = batch_norm2d(planes)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += x
        out = self.act(out)
        return out


class ConvBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = batch_norm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = batch_norm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = batch_norm2d(planes)
        self.conv_shortcut = nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.bn_shortcut = batch_norm2d(planes)
        self.act = nn.ReLU()

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.bn_shortcut(self.conv_shortcut(x))
        out = self.act(out)
        return out


class ResNet(nn.Module):
    def __init__(self, config, num_classes=100):
        super(ResNet, self).__init__()
        self.in_planes = 16

        self.features = False

        self.padd = nn.ZeroPad2d(3)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=0, bias=False)
        self.bn1 = batch_norm2d(16)

        unzip_info = list(zip(*config))
        num_layers = unzip_info[0]
        filters = unzip_info[1]
        strides = unzip_info[2]
        self.layer1 = self._make_layer(num_layers[0], filters[0], strides[0])
        self.layer2 = self._make_layer(num_layers[1], filters[1], strides[1])
        self.layer3 = self._make_layer(num_layers[2], filters[2], strides[2])
        self.bn2 = batch_norm2d(self.in_planes)
        # self.linear = nn.Linear(filters[2], num_classes)
        self.act = nn.ReLU()
        self.apply(_weights_init)

    def _make_layer(self, num_layers, planes, stride):
        layers = nn.ModuleList()
        layers.append(ConvBlock(self.in_planes, planes, stride))
        self.in_planes = planes
        for layer in range(num_layers - 1):
            layers.append(IdentityBlock(planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.padd(x)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.bn2(out)
        if self.features:
            features = out
        else:
            features = None

        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)

        return out, features


def create_model(use_selfatt=False, use_fc=False, dropout=None, stage1_weights=False, dataset=None, log_dir=None, test=False, *args, num_classes=10):
    print('Loading Scratch ResNet 32 Feature Model.')
    resnet32 = ResNet([(5, 16, 1), (5, 32, 2), (5, 64, 2)], num_classes)
    return resnet32


        
# def create_model(use_selfatt=False, use_fc=False, dropout=None, stage1_weights=False, dataset=None, log_dir=None, test=False, *args):
    
#     print('Loading Scratch ResNet 50 Feature Model.')
#     resnet50 = ResNet(Bottleneck, [3, 4, 6, 3], use_modulatedatt=use_selfatt, use_fc=use_fc, dropout=None)

#     if not test:
#         if stage1_weights:
#             assert(dataset)
#             print('Loading %s Stage 1 ResNet 10 Weights.' % dataset)
#             if log_dir is not None:
#                 # subdir = log_dir.strip('/').split('/')[-1]
#                 # subdir = subdir.replace('stage2', 'stage1')
#                 # weight_dir = path.join('/'.join(log_dir.split('/')[:-1]), subdir)
#                 weight_dir = path.join('/'.join(log_dir.split('/')[:-1]), 'stage1')
#             else:
#                 weight_dir = './logs/%s/stage1' % dataset
#             print('==> Loading weights from %s' % weight_dir)
#             resnet50 = init_weights(model=resnet50,
#                                     weights_path=path.join(weight_dir, 'final_model_checkpoint.pth'))
#         else:
#             print('No Pretrained Weights For Feature Model.')

#     return resnet50


# if __name__ == "__main__":
#     import torch
#     #from pytorch_model_summary import summary
#     resnet32 = ResNet([(5, 16, 1), (5, 32, 2), (5, 64, 2)], num_classes)
#     net = resnet32(10)
#     rand_inp = torch.rand((1, 3, 32, 32))
#     #summary(net, rand_inp, print_summary=True, max_depth=3, show_parent_layers=True)
