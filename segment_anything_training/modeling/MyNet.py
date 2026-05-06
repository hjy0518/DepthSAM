import torch
import torch.nn as nn
from .common import LayerNorm2d
import torch.nn.functional as F
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )

def conv1x1(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=has_bias)


def conv1x1_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv1x1(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )

from einops import rearrange


def custom_complex_normalization(input_tensor, dim=-1):
    real_part = input_tensor.real
    imag_part = input_tensor.imag
    norm_real = F.softmax(real_part, dim=dim)
    norm_imag = F.softmax(imag_part, dim=dim)

    normalized_tensor = torch.complex(norm_real, norm_imag)

    return normalized_tensor

class Attention_SD(nn.Module):
    def __init__(self, dim, num_heads=2):
        super(Attention_SD, self).__init__()
        self.num_heads = num_heads

        self.qkv1conv_1 = conv1x1(dim,dim)
        self.qkv1conv_3 = conv1x1(dim,dim)
        self.qkv1conv_5 = conv1x1(dim,dim)

        self.qm = conv1x1(dim,dim)
        self.km = conv1x1(dim,dim)
        self.vm = conv1x1(dim,dim)

        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.temperatured = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.project_out = conv3x3_bn_relu(dim * 4, dim)

        self.project_outr = nn.Conv2d(dim * 2, dim, kernel_size=1)

        self.weight = nn.Sequential(
            nn.Conv2d(dim, dim // 16, 1, bias=True),
            nn.BatchNorm2d(dim // 16),
            nn.ReLU(True),
            nn.Conv2d(dim // 16, dim, 1, bias=True),
            nn.Sigmoid())
        self.weightd = nn.Sequential(
            nn.Conv2d(dim, dim // 16, 1, bias=True),
            nn.BatchNorm2d(dim // 16),
            nn.ReLU(True),
            nn.Conv2d(dim // 16, dim, 1, bias=True),
            nn.Sigmoid())

        self.project_outd = nn.Conv2d(dim * 2, dim, kernel_size=1)


    def forward(self, x,d):

        b, c, h, w = x.shape
        q_s = self.qkv1conv_5(x)
        k_s = self.qkv1conv_3(x)
        v_s = self.qkv1conv_1(x)
        q_s = torch.fft.fft2(q_s.float())
        k_s = torch.fft.fft2(k_s.float())
        v_s = torch.fft.fft2(v_s.float())
        q_s = rearrange(q_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_s = rearrange(k_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_s = rearrange(v_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q_s = torch.nn.functional.normalize(q_s, dim=-1)
        k_s = torch.nn.functional.normalize(k_s, dim=-1)
        attn_s = (q_s @ k_s.transpose(-2, -1)) * self.temperature
        attn_s = custom_complex_normalization(attn_s, dim=-1)
        outr0 =  torch.abs(torch.fft.ifft2( attn_s @ v_s))
        attn_s = torch.abs(torch.fft.ifft2(attn_s))


        dq_s = self.qm(d)
        dk_s = self.km(d)
        dv_s = self.vm(d)
        dq_s = rearrange(dq_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        dk_s = rearrange(dk_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        dv_s = rearrange(dv_s, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        dq_s = torch.nn.functional.normalize(dq_s, dim=-1)
        dk_s = torch.nn.functional.normalize(dk_s, dim=-1)
        dattn_s = (dq_s @ dk_s.transpose(-2, -1)) * self.temperatured
        dattn_s = torch.softmax(dattn_s, dim=-1)
        outd0 = dattn_s @ dv_s
        dattn_s = torch.fft.fft2(dattn_s.float())

        outr = torch.abs(torch.fft.ifft2(dattn_s @ v_s))
        outd = attn_s @ dv_s

        outd = rearrange(outd, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        outr = rearrange(outr, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        outd0 = rearrange(outd0, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        outr0 = rearrange(outr0, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)


        out_f_lr = torch.abs(torch.fft.ifft2(self.weight(torch.fft.fft2(x.float()).real) * torch.fft.fft2(x.float())))
        outr = self.project_outr(torch.cat((outr, out_f_lr), 1))

        out_f_ld = torch.abs(torch.fft.ifft2(self.weightd(torch.fft.fft2(d.float()).real) * torch.fft.fft2(d.float())))
        outd = self.project_outd(torch.cat((outd, out_f_ld), 1))

        out = self.project_out(torch.cat((outr,outr0,outd,outd0), 1))


        return out

class FM(nn.Module):
    def __init__(self, dim,oup):
        super(FM, self).__init__()
        self.dim = oup
        self.conv_n = BasicConv2d(dim, self.dim, kernel_size=1, stride=1)
        self.asd = Attention_SD(self.dim)
        self.ddd = DWConv(self.dim)

    def forward(self,x):
        x = self.conv_n(x)
        out = self.asd(x,x)
        out = self.ddd(out)
        return out

class MEF(nn.Module):
    def __init__(self, in1,in2):
        super(MEF, self).__init__()
        self.fm = FM(in1 + in1,in1)
        self.fm1 = nn.Sequential(
            nn.ConvTranspose2d(in2, in1, kernel_size=2, stride=2),
            LayerNorm2d(in1),
            nn.GELU()
        )
    def forward(self, in1, in2=None):
        in2 = self.fm1(in2)
        x = torch.cat((in1, in2), 1)
        out = self.fm(x)
        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Decode(nn.Module):
    def __init__(self, in1,in2,in3,in4):
        super(Decode, self).__init__()
        # self.in3 = in3
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.premp = nn.Sequential(
            nn.ConvTranspose2d(in1, in1//2, kernel_size=2, stride=2),
            LayerNorm2d(in1//2),
            nn.GELU(),
            nn.Conv2d(in_channels=in1//2, out_channels=1, kernel_size=3, padding=1, bias=True),
        )


        self.fm1 = MEF(in1,in2)
        self.fm2 = MEF(in2,in3)
        self.fm3 = MEF(in3,in4)
    def forward(self,x1,x2,x3,x4):

        br3 = self.fm3(x3, x4)
        br2 = self.fm2(x2, br3)
        br1 = self.fm1(x1, br2)
        out = self.premp(br1)
        return out, br1


class DWConv(nn.Module):
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dim = dim//4
        self.dw2 = nn.Conv2d(self.dim, self.dim, kernel_size=7, padding=3, groups=self.dim)
        self.dw1 = nn.Conv2d(dim, self.dim, kernel_size=5, padding=2, groups=self.dim)
        self.dw3 = nn.Conv2d(self.dim, self.dim, kernel_size=3, padding=1, groups=self.dim)
        self.dw4 = nn.Conv2d(self.dim, self.dim, kernel_size=1)

        self.conv_end = conv1x1_bn_relu(dim,dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.dw1(x)
        x2 = self.dw2(x1) + x1
        x3 = self.dw3(x2) + x2
        x4 = self.dw4(x3) + x3

        x = self.conv_end(torch.cat((x1,x2,x3,x4),1))
        return x
