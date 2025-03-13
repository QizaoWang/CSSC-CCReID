"""
@author:  Qizao Wang
@contact: qzwang22@m.fudan.edu.cn

ICASSP 2025 paper: Content and Salient Semantics Collaboration for Cloth-Changing Person Re-Identification
URL: arxiv.org/abs/2405.16597
GitHub: https://github.com/QizaoWang/CSSC-CCReID
"""

import copy

import torch
import torchvision
import torch.nn as nn


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class SMR(nn.Module):
    def __init__(self, pool_type='avg', part_dim=256, part_num=8, reduction=16, feature_dim=-1, num_classes=-1):
        super().__init__()
        self.pool_type = pool_type
        self.part_dim = part_dim
        self.part_num = part_num
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        self.bottleneck = nn.BatchNorm1d(self.feature_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.pool = nn.AdaptiveAvgPool2d(1) if self.pool_type == 'avg' else nn.AdaptiveMaxPool2d(1)
        self.l_conv_list = nn.ModuleList()
        for i in range(self.part_num):
            self.l_conv_list.append(nn.Sequential(nn.Linear(self.feature_dim, self.part_dim, bias=False),
                                                  nn.BatchNorm1d(self.part_dim)))

        embed_dim = self.part_num * self.part_dim + self.feature_dim
        self.classifier = nn.Linear(embed_dim, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.refinement = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim // reduction, self.feature_dim, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x):
        g_feat = self.pool(x).flatten(1)

        if self.training:
            g_feat_bn = self.bottleneck(g_feat)
            l_feat_list = []
            part_len = x.shape[2] // self.part_num
            for i in range(self.part_num):
                l_feat = self.pool(x[:, :, i * part_len: (i + 1) * part_len, :]).flatten(1)
                l_feat_conv = self.l_conv_list[i](l_feat)
                l_feat_list.append(l_feat_conv)

            feat_bn = torch.cat([g_feat_bn, torch.cat(l_feat_list, dim=-1)], dim=-1)
            y = self.classifier(feat_bn)

        x_refined = self.refinement(g_feat).unsqueeze(-1).unsqueeze(-1) * x

        if self.training:
            return x_refined, g_feat, y
        else:
            return x_refined


class CSSC(nn.Module):
    def __init__(self, last_stride=1, pretrain=True, num_classes=None):
        super().__init__()
        resnet = getattr(torchvision.models, 'resnet50')(pretrained=pretrain)
        resnet.layer4[0].downsample[0].stride = (last_stride, last_stride)
        resnet.layer4[0].conv2.stride = (last_stride, last_stride)
        self.backbone_before_layer4 = nn.Sequential(*list(resnet.children())[:-3])

        self.num_classes = num_classes
        feature_dim = 2048
        self.feature_dim = feature_dim

        self.backbone_layer41_branch1 = resnet.layer4[0]
        self.backbone_layer42_branch1 = resnet.layer4[1]
        self.backbone_layer41_branch2 = copy.deepcopy(resnet.layer4[0])
        self.backbone_layer42_branch2 = copy.deepcopy(resnet.layer4[1])
        self.backbone_layer43 = resnet.layer4[2]

        self.smr_c_branch1 = SMR(pool_type='avg', feature_dim=feature_dim, num_classes=num_classes)
        self.smr_s_branch1 = SMR(pool_type='max', feature_dim=feature_dim, num_classes=num_classes)
        self.smr_s_branch2 = SMR(pool_type='max', feature_dim=feature_dim, num_classes=num_classes)
        self.smr_c_branch2 = SMR(pool_type='avg', feature_dim=feature_dim, num_classes=num_classes)

        self.pool = nn.AdaptiveMaxPool2d(1)
        self.bottleneck = nn.BatchNorm1d(self.feature_dim)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier = nn.Linear(self.feature_dim, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        x = self.backbone_before_layer4(x)

        # branch 1
        x_c_branch1 = self.backbone_layer41_branch1(x)
        if not self.training:
            x_c_branch1 = self.smr_c_branch1(x_c_branch1)
        else:
            x_c_branch1, feat_c_branch1, y_c_branch1 = self.smr_c_branch1(x_c_branch1)

        x_cs_branch1 = self.backbone_layer42_branch1(x_c_branch1)
        if not self.training:
            x_cs_branch1 = self.smr_s_branch1(x_cs_branch1)
        else:
            x_cs_branch1, feat_cs_branch1, y_cs_branch1 = self.smr_s_branch1(x_cs_branch1)

        # branch 2
        x_s_branch2 = self.backbone_layer41_branch2(x)
        if not self.training:
            x_s_branch2 = self.smr_s_branch2(x_s_branch2)
        else:
            x_s_branch2, feat_s_branch2, y_s_branch2 = self.smr_s_branch2(x_s_branch2)

        x_sc_branch2 = self.backbone_layer42_branch2(x_s_branch2)
        if not self.training:
            x_sc_branch2 = self.smr_c_branch2(x_sc_branch2)
        else:
            x_sc_branch2, feat_sc_branch2, y_sc_branch2 = self.smr_c_branch2(x_sc_branch2)

        x_cssc = self.backbone_layer43(x_cs_branch1 + x_sc_branch2)
        feat_cssc = self.pool(x_cssc).flatten(1)

        if self.training:
            feat_cssc_bn = self.bottleneck(feat_cssc)
            y_cssc = self.classifier(feat_cssc_bn)
            return [feat_c_branch1, feat_cs_branch1, feat_s_branch2, feat_sc_branch2, feat_cssc], \
                [y_c_branch1, y_cs_branch1, y_s_branch2, y_sc_branch2, y_cssc]
        else:
            return feat_cssc
