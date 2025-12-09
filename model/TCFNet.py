import torch
import torch.nn as nn
import torch.nn.functional as F
from model.resnet_aspp import ResNet_ASPP
from model.pvtv2_encoder import pvt_v2_b4
from model.aspp import ASPP

def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()

class CFM(nn.Module):
    def __init__(self):
        super(CFM, self).__init__()
        self.conv1h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1h   = nn.BatchNorm2d(64)
        self.conv2h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2h   = nn.BatchNorm2d(64)
        self.conv3h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3h   = nn.BatchNorm2d(64)
        self.conv4h = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4h   = nn.BatchNorm2d(64)

        self.conv1v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn1v   = nn.BatchNorm2d(64)
        self.conv2v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2v   = nn.BatchNorm2d(64)
        self.conv3v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3v   = nn.BatchNorm2d(64)
        self.conv4v = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4v   = nn.BatchNorm2d(64)

    def forward(self, left, down):
        if down.size()[2:] != left.size()[2:]:
            down = F.interpolate(down, size=left.size()[2:], mode='bilinear')
        out1h = F.relu(self.bn1h(self.conv1h(left)), inplace=True)
        out1v = F.relu(self.bn1v(self.conv1v(down)), inplace=True)
        out2h = F.relu(self.bn2h(self.conv2h(out1h)), inplace=True)
        out2v = F.relu(self.bn2v(self.conv2v(out1v)), inplace=True)
        fuse = out2h*out2v
        out3h = F.relu(self.bn3h(self.conv3h(fuse)), inplace=True)+out1h
        out4h = F.relu(self.bn4h(self.conv4h(out3h)), inplace=True)
        out3v = F.relu(self.bn3v(self.conv3v(fuse)), inplace=True)+out1v
        out4v = F.relu(self.bn4v(self.conv4v(out3v)), inplace=True)
        return out4h, out4v

    def initialize(self):
        weight_init(self)

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.cfm45 = CFM()
        self.cfm34 = CFM()
        self.cfm23 = CFM()
        self.cfm12 = CFM()

        self.gate_weight = nn.Sequential(
            nn.Conv2d(64 * 2, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Sigmoid()
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

    def forward(self, out1h, out2h, out3h, out4h, out5v, fback=None):
        if fback is not None:
            fb_curr5 = F.interpolate(fback, size=out5v.shape[2:], mode='bilinear')
            concat_feat5 = torch.cat([out5v, fb_curr5], dim=1)
            gate5 = self.gate_weight(concat_feat5)
            fused_feat5 = self.conv_cat(out5v + gate5 * fb_curr5)

            fb_curr4 = F.interpolate(fback, size=out4h.shape[2:], mode='bilinear')
            concat_feat4 = torch.cat([out4h, fb_curr4], dim=1)
            gate4 = self.gate_weight(concat_feat4)
            fused_feat4 = self.conv_cat(out4h + gate4 * fb_curr4)

            fb_curr3 = F.interpolate(fback, size=out3h.shape[2:], mode='bilinear')
            concat_feat3 = torch.cat([out3h, fb_curr3], dim=1)
            gate3 = self.gate_weight(concat_feat3)
            fused_feat3 = self.conv_cat(out3h + gate3 * fb_curr3)

            fb_curr2 = F.interpolate(fback, size=out2h.shape[2:], mode='bilinear')
            concat_feat2 = torch.cat([out2h, fb_curr2], dim=1)
            gate2 = self.gate_weight(concat_feat2)
            fused_feat2 = self.conv_cat(out2h + gate2 * fb_curr2)

            fb_curr1 = F.interpolate(fback, size=out1h.shape[2:], mode='bilinear')
            concat_feat1 = torch.cat([out1h, fb_curr1], dim=1)
            gate1 = self.gate_weight(concat_feat1)
            fused_feat1 = self.conv_cat(out1h + gate1 * fb_curr1)

            out5v = fused_feat5
            out4h, out4v = self.cfm45(fused_feat4, out5v)
            out3h, out3v = self.cfm34(fused_feat3, out4v)
            out2h, out2v = self.cfm23(fused_feat2, out3v)
            out1h, pred = self.cfm12(fused_feat1, out2v)
        else:
            out4h, out4v = self.cfm45(out4h, out5v)
            out3h, out3v = self.cfm34(out3h, out4v)
            out2h, out2v = self.cfm23(out2h, out3v)
            out1h, pred = self.cfm12(out1h, out2v)
        return out1h, out2h, out3h, out4h, out5v, pred

    def initialize(self):
        weight_init(self)

class ChannelAttention(nn.Module):
    def __init__(self, channel, ratio=2):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Linear(channel, channel // ratio),nn.ReLU(True),nn.Linear(channel // ratio, channel),nn.Sigmoid())

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return y

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        weight_map = self.sigmoid(x)
        return weight_map

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class ATFIFM(nn.Module):
    def __init__(self, in_planes):
        super(ATFIFM, self).__init__()
        self.catconv1 = BasicConv2d(in_planes=in_planes * 2, out_planes=in_planes, kernel_size=3,padding=1, stride=1)
        self.catconv2 = BasicConv2d(in_planes=in_planes * 4, out_planes=in_planes, kernel_size=3,padding=1, stride=1)
        self.catconv3 = BasicConv2d(in_planes=in_planes * 2, out_planes=in_planes, kernel_size=3, padding=1, stride=1)
        self.catconv4 = BasicConv2d(in_planes=in_planes * 4, out_planes=in_planes, kernel_size=3, padding=1, stride=1)
        self.catconv5 = BasicConv2d(in_planes=in_planes, out_planes=in_planes, kernel_size=3, padding=1, stride=1)

        self.ca_rgb = ChannelAttention(in_planes)
        self.ca_flow = ChannelAttention(in_planes)
        self.ca_depth = ChannelAttention(in_planes)

        self.sa_rgb = SpatialAttention(kernel_size=7)
        self.sa_flow = SpatialAttention(kernel_size=7)
        self.sa_depth = SpatialAttention(kernel_size=7)

        self.shortcut1 = nn.Conv2d(in_planes * 4, in_planes, kernel_size=1)
        self.shortcut2 = nn.Conv2d(in_planes * 4, in_planes, kernel_size=1)

    def forward(self, input1, input2, input3):
        B, C, H, W = input1.size()
        P = H * W
        rgb_SA = self.sa_rgb(input1).view(B, -1, P)  # B * 1 * H * W
        flow_SA = self.sa_flow(input2).view(B, -1, P)
        depth_SA = self.sa_depth(input3).view(B, -1, P)

        rgb_CA = self.ca_rgb(input1).view(B, C, -1)  # B * C * 1 * 1
        flow_CA = self.ca_flow(input2).view(B, C, -1)
        depth_CA = self.ca_depth(input3).view(B, C, -1)

        rgb_M = torch.bmm(rgb_CA, rgb_SA).view(B, C, H, W)
        flow_M = torch.bmm(flow_CA, flow_SA).view(B, C, H, W)
        depth_M = torch.bmm(depth_CA, depth_SA).view(B, C, H, W)

        rgb_smAR = input1 * rgb_M + input1
        flow_smAR = input2 * flow_M + input2
        depth_smAR = input3 * depth_M + input3

        cat_out = self.catconv1(torch.cat([rgb_smAR, flow_smAR], dim=1))
        mul_out = rgb_smAR * flow_smAR
        sub_out = rgb_smAR - flow_smAR
        max_put = torch.maximum(rgb_smAR, flow_smAR)
        combined = self.catconv2(torch.cat([cat_out, mul_out, sub_out, max_put], dim=1))
        shortcut = self.shortcut1(torch.cat([rgb_smAR, flow_smAR, depth_smAR, max_put], dim=1))
        interactive_out = combined + shortcut

        cat_out1 = self.catconv3(torch.cat([rgb_smAR, depth_smAR], dim=1))
        mul_out1 = rgb_smAR * depth_smAR
        sub_out1 = rgb_smAR - depth_smAR
        max_put1 = torch.maximum(rgb_smAR, depth_smAR)
        combined1 = self.catconv4(torch.cat([cat_out1, mul_out1, sub_out1, max_put1], dim=1))
        shortcut1 = self.shortcut2(torch.cat([rgb_smAR, flow_smAR, depth_smAR, max_put1], dim=1))
        interactive_out1 = combined1 + shortcut1

        cat = self.catconv5(interactive_out+interactive_out1)
        return cat

class UIM_fuse(nn.Module):
    def __init__(self, in_planes):
        super(UIM_fuse, self).__init__()
        self.catconv1 = BasicConv2d(in_planes=in_planes * 2, out_planes=in_planes, kernel_size=3, padding=1, stride=1)
        self.catconv2 = BasicConv2d(in_planes=in_planes * 4, out_planes=in_planes, kernel_size=3, padding=1, stride=1)

    def forward(self, input1, input2):
        cat_out = self.catconv1(torch.cat([input1,input2], dim=1))
        mul_out = input1 * input2
        sub_out = input1 - input2
        max_put = torch.maximum(input1, input2)
        interactive_out = self.catconv2(torch.cat([cat_out, mul_out, sub_out, max_put], dim=1))

        return interactive_out

class Model(nn.Module):
    def __init__(self, inchannels, mode):
        super(Model, self).__init__()
        self.ImageBone = pvt_v2_b4()
        self.FlowBone = ResNet_ASPP(inchannels, 1, 16, 'resnet34')
        self.DepthBone = ResNet_ASPP(inchannels, 1, 16, 'resnet34')

        self.aspp = ASPP(512, 256, [1, 6, 12, 18])

        self.Channel_align1 = nn.Sequential(nn.Conv2d(64, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.Channel_align2 = nn.Sequential(nn.Conv2d(128, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.Channel_align3 = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.Channel_align4 = nn.Sequential(nn.Conv2d(512, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.Channel_align5 = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.T_layer = nn.Sequential(nn.Conv2d(in_channels=320, out_channels=256, kernel_size=3, padding=1, bias=True), nn.ReLU(inplace=True))

        self.atfifm1 = ATFIFM(64)
        self.atfifm2 = ATFIFM(64)
        self.atfifm3 = ATFIFM(64)
        self.atfifm4 = ATFIFM(64)
        self.atfifm5 = ATFIFM(64)

        self.fuse_last_auxiliary = UIM_fuse(1)
        self.fuse_last = UIM_fuse(1)

        self.decoder1 = Decoder()
        self.decoder2 = Decoder()

        self.linearp1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearp2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.linearr1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linearr5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

        self.linearr_atfifm1 = nn.Conv2d(64, 1, 1)
        self.linearr_atfifm2 = nn.Conv2d(64, 1, 1)
        self.linearr_atfifm3 = nn.Conv2d(64, 1, 1)
        self.linearr_atfifm4 = nn.Conv2d(64, 1, 1)
        self.linearr_atfifm5 = nn.Conv2d(64, 1, 1)

        self.linearr_deco1_1 = nn.Conv2d(64, 1, 1)
        self.linearr_deco1_2 = nn.Conv2d(64, 1, 1)
        self.linearr_deco1_3 = nn.Conv2d(64, 1, 1)
        self.linearr_deco1_4 = nn.Conv2d(64, 1, 1)
        self.linearr_deco1_5 = nn.Conv2d(64, 1, 1)

        self.last_conv_rgb = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1, stride=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if mode == 'train':
            self.ImageBone._load_pretrained_model('./checkpoints/pvt_v2_b4.pth')
            self.FlowBone.backbone_features._load_pretrained_model('./checkpoints/resnet34-333f7ec4.pth')
            self.DepthBone.backbone_features._load_pretrained_model('./checkpoints/resnet34-333f7ec4.pth')

    def decoder_attention_module(self, img_feat, flow_map, depth, channels):
        self.depth_fc = nn.Linear(channels, channels).cuda()

        flow_attention = torch.sigmoid(flow_map)
        flow_weighted_feat = img_feat * flow_attention

        depth_pool = torch.mean(depth, dim=[2, 3], keepdim=True)
        depth_attention = torch.sigmoid(self.depth_fc(depth_pool))
        depth_weighted_feat = img_feat * depth_attention

        final_feat = img_feat + flow_weighted_feat + depth_weighted_feat

        return final_feat, flow_map, depth

    def load_pretrain_model(self, model_path):
        pretrain_dict = torch.load(model_path)
        model_dict = {}
        state_dict = self.state_dict()
        for k, v in pretrain_dict.items():
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        self.load_state_dict(state_dict)

    def forward(self, image, flow, depth):
        img_layer4_feat, img_layer3_feat, img_layer2_feat, img_layer1_feat = self.ImageBone(image)
        img_aspp_feat = self.aspp(img_layer4_feat)
        img_layer3_feat = self.T_layer(img_layer3_feat)
        img_layer4_feat = F.interpolate(img_layer4_feat, img_layer3_feat.shape[2:], mode='bilinear', align_corners=True)
        img_aspp_feat = F.interpolate(img_aspp_feat, img_layer3_feat.shape[2:], mode='bilinear', align_corners=True)
        course_img = self.last_conv_rgb(img_aspp_feat)

        flow_layer4_feat, flow_layer1_feat, flow_conv1_feat, flow_layer2_feat, flow_layer3_feat, flow_aspp_feat, course_flowmap = self.FlowBone(flow)
        depth_layer4_feat, depth_layer1_feat, depth_conv1_feat, depth_layer2_feat, depth_layer3_feat, depth_aspp_feat, course_depthmap = self.DepthBone(depth)

        course_img,course_flowmap,course_depthmap = self.decoder_attention_module(course_img,course_flowmap,course_depthmap,1)

        img_layer1_feat, img_layer2_feat, img_layer3_feat, img_layer4_feat, img_aspp_feat = self.Channel_align1(img_layer1_feat), self.Channel_align2(img_layer2_feat), self.Channel_align3(img_layer3_feat), self.Channel_align4(img_layer4_feat), self.Channel_align5(img_aspp_feat)
        flow_layer1_feat, flow_layer2_feat, flow_layer3_feat, flow_layer4_feat, flow_aspp_feat = self.Channel_align1(flow_layer1_feat), self.Channel_align2(flow_layer2_feat), self.Channel_align3(flow_layer3_feat), self.Channel_align4(flow_layer4_feat), self.Channel_align5(flow_aspp_feat)
        depth_layer1_feat, depth_layer2_feat, depth_layer3_feat, depth_layer4_feat, depth_aspp_feat = self.Channel_align1(depth_layer1_feat), self.Channel_align2(depth_layer2_feat), self.Channel_align3(depth_layer3_feat), self.Channel_align4(depth_layer4_feat), self.Channel_align5(depth_aspp_feat)

        out_layer1 = self.atfifm1(img_layer1_feat, flow_layer1_feat, depth_layer1_feat)
        out_layer2 = self.atfifm2(img_layer2_feat, flow_layer2_feat, depth_layer2_feat)
        out_layer3 = self.atfifm3(img_layer3_feat, flow_layer3_feat, depth_layer3_feat)
        out_layer4 = self.atfifm4(img_layer4_feat, flow_layer4_feat, depth_layer4_feat)
        out_layer5 = self.atfifm5(img_aspp_feat, flow_aspp_feat, depth_aspp_feat)

        shape = image.size()[2:]
        out_atfifm_layer1 = F.interpolate(self.linearr_atfifm1(out_layer1), size=shape, mode='bilinear')
        out_atfifm_layer2 = F.interpolate(self.linearr_atfifm2(out_layer2), size=shape, mode='bilinear')
        out_atfifm_layer3 = F.interpolate(self.linearr_atfifm3(out_layer3), size=shape, mode='bilinear')
        out_atfifm_layer4 = F.interpolate(self.linearr_atfifm4(out_layer4), size=shape, mode='bilinear')
        out_atfifm_layer5 = F.interpolate(self.linearr_atfifm5(out_layer5), size=shape, mode='bilinear')

        out_dec1_1, out_dec1_2, out_dec1_3, out_dec1_4, out_dec1_5, pred1 = self.decoder1(out_layer1, out_layer2, out_layer3, out_layer4, out_layer5)

        out_deco1_layer1 = F.interpolate(self.linearr_deco1_1(out_dec1_1), size=shape, mode='bilinear')
        out_deco1_layer2 = F.interpolate(self.linearr_deco1_2(out_dec1_2), size=shape, mode='bilinear')
        out_deco1_layer3 = F.interpolate(self.linearr_deco1_3(out_dec1_3), size=shape, mode='bilinear')
        out_deco1_layer4 = F.interpolate(self.linearr_deco1_4(out_dec1_4), size=shape, mode='bilinear')
        out_deco1_layer5 = F.interpolate(self.linearr_deco1_5(out_dec1_5), size=shape, mode='bilinear')

        out_dec2_1, out_dec2_2, out_dec2_3, out_dec2_4, out_dec2_5, pred2 = self.decoder2(out_dec1_1, out_dec1_2, out_dec1_3, out_dec1_4, out_dec1_5, pred1)

        pred11 = F.interpolate(self.linearp1(pred1), size=shape, mode='bilinear', align_corners=True)
        pred22 = F.interpolate(self.linearp2(pred2), size=shape, mode='bilinear', align_corners=True)

        course_img = F.interpolate(course_img, size=shape, mode='bilinear', align_corners=True)
        course_flowmap = F.interpolate(course_flowmap, size=shape, mode='bilinear', align_corners=True)
        course_depthmap = F.interpolate(course_depthmap, size=shape, mode='bilinear', align_corners=True)

        out1 = F.interpolate(self.linearr1(out_dec2_1), size=shape, mode='bilinear', align_corners=True)
        out2 = F.interpolate(self.linearr2(out_dec2_2), size=shape, mode='bilinear', align_corners=True)
        out3 = F.interpolate(self.linearr3(out_dec2_3), size=shape, mode='bilinear', align_corners=True)
        out4 = F.interpolate(self.linearr4(out_dec2_4), size=shape, mode='bilinear', align_corners=True)
        out5 = F.interpolate(self.linearr5(out_dec2_5), size=shape, mode='bilinear', align_corners=True)

        return (pred11, pred22, out1, out2, out3, out4, out5, course_img, course_flowmap, course_depthmap,
                out_atfifm_layer1, out_atfifm_layer2, out_atfifm_layer3, out_atfifm_layer4, out_atfifm_layer5,
                out_deco1_layer1, out_deco1_layer2, out_deco1_layer3, out_deco1_layer4, out_deco1_layer5)

