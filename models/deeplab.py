import torch
import torch.nn as nn
import torch.nn.functional as F
from models.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from models.aspp import build_aspp
from models.decoder import build_decoder
from models.backbone import build_backbone

class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=21,
                 sync_bn=True, freeze_bn=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat, output_by_classifier, last_layer_feature_by_classifier = self.backbone(input)
        # x is last output, low_level_feature is shallow layer output
        # if input size is (1, 3, 256, 256)

        # print(x.size())
        # # torch.Size([1, 2048, 16, 16])
        # print(low_level_feat.size())
        # # torch.Size([1, 128, 64, 64])

        x = self.aspp(x)
        # print(x.size())
        # torch.Size([1, 256, 16, 16])

        x = self.decoder(x, low_level_feat)
        # print(x.size())
        # torch.Size([1, 2, 64, 64])

        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        # print(x.size())
        # exit(0)
        # 修改输出后的尺寸
        # torch.Size([16, 3, 64, 64])
        # torch.Size([16, 3, 256, 256])


        return x, output_by_classifier, last_layer_feature_by_classifier

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

if __name__ == "__main__":
    model = DeepLab(backbone='xception', output_stride=16, num_classes=2)
    model.eval()
    # print(model)
    input = torch.rand(16, 3, 256, 256)
    output, output_by_classifier, last_layer_feature_by_classifier = model(input)
    print(output.size())
    print(output_by_classifier.size())
    print(last_layer_feature_by_classifier.size())
    # torch.Size([16, 2, 256, 256])
    # torch.Size([16, 2])
    # torch.Size([16, 2048])


