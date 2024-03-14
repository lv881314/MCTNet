import torch
from torch import nn
import torch.nn.functional as F

import model.resnet as models
from model.transformer.transformer import Transformer
from model.transformer.position_encoding import PositionEmbeddingSine


class GroupBatchnorm2d(nn.Module):
    def __init__(self, c_num: int,
                 group_num: int = 16,
                 eps: float = 1e-10
                 ):
        super(GroupBatchnorm2d, self).__init__()
        assert c_num >= group_num
        self.group_num = group_num
        self.gamma = nn.Parameter(torch.randn(c_num, 1, 1))
        self.beta = nn.Parameter(torch.zeros(c_num, 1, 1))
        self.eps = eps

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, self.group_num, -1)
        mean = x.mean(dim=2, keepdim=True)
        std = x.std(dim=2, keepdim=True)
        x = (x - mean) / (std + self.eps)
        x = x.view(N, C, H, W)
        return x * self.gamma + self.beta


class SRU(nn.Module):
    def __init__(self,
                 oup_channels: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5
                 ):
        super().__init__()

        self.gn = GroupBatchnorm2d(oup_channels, group_num=group_num)
        self.gate_treshold = gate_treshold
        self.sigomid = nn.Sigmoid()

    def forward(self, x):
        gn_x = self.gn(x)
        w_gamma = self.gn.gamma / sum(self.gn.gamma)
        reweigts = self.sigomid(gn_x * w_gamma)
        # Gate
        info_mask = reweigts >= self.gate_treshold
        noninfo_mask = reweigts < self.gate_treshold
        x_1 = info_mask * x
        x_2 = noninfo_mask * x
        x = self.reconstruct(x_1, x_2)
        return x

    def reconstruct(self, x_1, x_2):
        x_11, x_12 = torch.split(x_1, x_1.size(1) // 2, dim=1)
        x_21, x_22 = torch.split(x_2, x_2.size(1) // 2, dim=1)
        return torch.cat([x_11 + x_22, x_12 + x_21], dim=1)


class CRU(nn.Module):
    '''
    alpha: 0<alpha<1
    '''

    def __init__(self,
                 op_channel: int,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.up_channel = up_channel = int(alpha * op_channel)
        self.low_channel = low_channel = op_channel - up_channel
        self.squeeze1 = nn.Conv2d(up_channel, up_channel // squeeze_radio, kernel_size=1, bias=False)
        self.squeeze2 = nn.Conv2d(low_channel, low_channel // squeeze_radio, kernel_size=1, bias=False)
        # up
        self.GWC = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=group_kernel_size, stride=1,
                             padding=group_kernel_size // 2, groups=group_size)
        self.PWC1 = nn.Conv2d(up_channel // squeeze_radio, op_channel, kernel_size=1, bias=False)
        # low
        self.PWC2 = nn.Conv2d(low_channel // squeeze_radio, op_channel - low_channel // squeeze_radio, kernel_size=1,
                              bias=False)
        self.advavg = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # Split
        up, low = torch.split(x, [self.up_channel, self.low_channel], dim=1)
        up, low = self.squeeze1(up), self.squeeze2(low)
        # Transform
        Y1 = self.GWC(up) + self.PWC1(up)
        Y2 = torch.cat([self.PWC2(low), low], dim=1)
        # Fuse
        out = torch.cat([Y1, Y2], dim=1)
        out = F.softmax(self.advavg(out), dim=1) * out
        out1, out2 = torch.split(out, out.size(1) // 2, dim=1)
        return out1 + out2


class ScConv(nn.Module):
    def __init__(self,
                 op_channel: int,
                 group_num: int = 16,
                 gate_treshold: float = 0.5,
                 alpha: float = 1 / 2,
                 squeeze_radio: int = 2,
                 group_size: int = 2,
                 group_kernel_size: int = 3,
                 ):
        super().__init__()
        self.SRU = SRU(op_channel,
                       group_num=group_num,
                       gate_treshold=gate_treshold)
        self.CRU = CRU(op_channel,
                       alpha=alpha,
                       squeeze_radio=squeeze_radio,
                       group_size=group_size,
                       group_kernel_size=group_kernel_size)

    def forward(self, x):
        x = self.SRU(x)
        x = self.CRU(x)
        return x

class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])
        self._init_weight()

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class TransformNet1(nn.Module):
    def __init__(self, layers=50, bins=(1, 3, 6, 8), dropout=0.2, classes=2, zoom_factor=8,
                 use_aspp=True, output_stride=8, criterion=nn.CrossEntropyLoss(ignore_index=255),
                 hidden_dim=512, BatchNorm=nn.BatchNorm2d, pretrained=True, ImLength=110):
        super(TransformNet1, self).__init__()
        assert layers in [50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_aspp = use_aspp
        self.criterion = criterion
        self.os = output_stride
        self.bins = bins
        models.BatchNorm = BatchNorm

        if layers == 50:
            resnet = models.resnet50(output_stride, BatchNorm, pretrained=pretrained)
        elif layers == 101:
            resnet = models.resnet101(output_stride, BatchNorm, pretrained=pretrained)
        else:
            resnet = models.resnet152(output_stride, BatchNorm, pretrained=pretrained)
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu,
                                    resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4

        self.feat_proj = nn.Conv2d(resnet.num_channels[-1], hidden_dim, kernel_size=1)

        self.ppm = PSPModule(sizes=(1, 3, 6, 8), dimension=2)
        self.ScConv = ScConv(512)

        self.pos_enc = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
        self.transformer = Transformer(d_model=hidden_dim, nhead=8, num_encoder_layers=0,
                                       num_decoder_layers=4, dim_feedforward=2048, dropout=0.1,
                                       activation="relu", normalize_before=False,
                                       return_intermediate_dec=False)

        self.query_embed = nn.Embedding(ImLength, hidden_dim)

        self.cls = nn.Sequential(
            nn.Conv2d(hidden_dim*(1+len(bins)), 512, kernel_size=3, padding=1, bias=False),
            BatchNorm(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(512, classes, kernel_size=1)
        )
        if self.training:
            self.aux = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                BatchNorm(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )
            self.tr_dec_aux1 = nn.Sequential(
                nn.Conv2d(hidden_dim*(1+len(bins)), 512, kernel_size=3, padding=1, bias=False),
                BatchNorm(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(512, classes, kernel_size=1)
            )

    def forward(self, x, y=None):
        x_size = x.size()

        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        x = self.layer0(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f3_aux = f3
        f4 = self.layer4(f3)
        #print("LLLLLLLLLLLL")
        #print(f4.size()) #torch.Size([6, 2048, 65, 65])
        proj_f4 = self.feat_proj(f4)
        #print(proj_f4.size()) #torch.Size([6, 512, 65, 65])
        proj_f4 = self.ScConv(proj_f4)
        #print(proj_f4.size()) #torch.Size([6, 512, 65, 65])
        spp = self.ppm(proj_f4)
        #print(spp.size())#torch.Size([6, 512, 110])

        proj_f4_half = F.interpolate(proj_f4, scale_factor=0.3, mode='bilinear', align_corners=True)
        #print(proj_f4_half.size()) # ([2, 512, 19, 19])
        pos = self.pos_enc(proj_f4_half)
        #print(pos.size()) #([2, 512, 19, 19])
        mask = torch.zeros(torch.max(proj_f4_half, dim=1)[0].size()).type(torch.BoolTensor).cuda()
        #print(mask.size()) #[2, 19, 19])

        tr_output, tr_aux1 = self.transformer(src=proj_f4_half, mask=mask, tgt=spp, query_embed=self.query_embed.weight,
                                             pos_embed=pos)

        bsf, cf, hf, wf = proj_f4.shape
        #print(tr_output.size()) [2, 512, 110])

        psp_idx = 0
        psp_cat = proj_f4
       # print(psp_cat.size()) #[2, 512, 65, 65])
        psp_cat_aux1 = proj_f4
        for i in self.bins:
            square = i**2
            pooled_output = tr_output[:,:,psp_idx:psp_idx+square].view(bsf, cf, i, i)
            pooled_resized_output = F.interpolate(pooled_output, size=proj_f4.size()[-2:], mode='bilinear', align_corners=True)
            psp_cat = torch.cat([psp_cat, pooled_resized_output], dim=1)

            if self.training:
                pooled_aux1_output = tr_aux1[:,:,psp_idx:psp_idx+square].view(bsf, cf, i, i)
                pooled_resized_aux1_output = F.interpolate(pooled_aux1_output, size=proj_f4.size()[-2:], mode='bilinear',
                                                          align_corners=True)
                psp_cat_aux1 = torch.cat([psp_cat_aux1, pooled_resized_aux1_output], dim=1)

            psp_idx = psp_idx + square

        x = self.cls(psp_cat)
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            aux = self.aux(f3_aux)
            tr_dec_aux1 = self.tr_dec_aux1(psp_cat_aux1)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
                tr_dec_aux1 = F.interpolate(tr_dec_aux1, size=(h, w), mode='bilinear', align_corners=True)
            main_loss = self.criterion(x, y)
            aux_loss = self.criterion(aux, y)
            tr_aux1_loss = self.criterion(tr_dec_aux1, y)
            return x.max(1)[1], main_loss, aux_loss, 0.3 * tr_aux1_loss
        else:
            return x
