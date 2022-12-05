import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.aspp import build_aspp
from models.decoder import build_decoder
from models.backbone import build_backbone
from torchvision import transforms

class DeepLab_Genera_Stream(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab_Genera_Stream, self).__init__()

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.Upper_stream = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn
        # 模糊半径越大, 正态分布标准差越大, 图像就越模糊
        # transform_1 = transforms.GaussianBlur(21, 10)
        # transform_2 = transforms.GaussianBlur(101, 10)
        # transform_3 = transforms.GaussianBlur(101, 100)
        self.transform = transforms.GaussianBlur(101, 10)

    def forward(self, input):
        # x, low_level_feat, output_by_classifier, last_layer_feature_by_classifier = self.backbone(input)
        x, low_level_feat = self.backbone(input)

        input = self.transform(input)
        y, low_level_feat = self.Upper_stream(input)
        x, output_by_classifier, last_layer_feature_by_classifier = self.backbone.classifier(x, y)
        # x is last output, low_level_feature is shallow layer output
        # if input size is (1, 3, 256, 256)

        # print(x.size())
        # print(low_level_feat.size())
        # torch.Size([16, 2048, 16, 16])
        # torch.Size([16, 128, 64, 64])

        x = self.aspp(x)
        # print(x.size())
        # torch.Size([16, 256, 16, 16])

        x = self.decoder(x, low_level_feat)
        # print(x.size())
        # exit(0)
        # torch.Size([16, 3, 64, 64])

        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        # print(x.size())
        # exit(0)
        # torch.Size([16, 3, 256, 256])

        # 此处x为输出的mask图像，ouput_by_class是分类结果
        return x, output_by_classifier, last_layer_feature_by_classifier

if __name__ == "__main__":
    model = DeepLab_Genera_Stream(backbone='xception', output_stride=16, num_classes=2)
    model.eval()
    # print(model)
    input = torch.rand(16, 3, 256, 256)
    output, output_by_classifier, last_layer_feature_by_classifier = model(input)
    print(output.size())
    print(output_by_classifier.size())
    print(last_layer_feature_by_classifier.size())
    # torch.Size([16, 3, 256, 256])
    # torch.Size([16, 2])
    # torch.Size([16, 2048])



