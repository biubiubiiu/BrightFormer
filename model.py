import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import SqueezeExcitation


class ConvenientLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, -1).transpose(-2, -1).contiguous()
        x = self.norm(x).transpose(-2, -1).contiguous()
        x = x.reshape(b, c, h, w)
        return x


class CrossConv(nn.Module):
    def __init__(self, channels, M):
        super().__init__()
        self.hconv = nn.Conv2d(channels, channels, kernel_size=(1, M), stride=1, padding=(0, 1))
        self.vconv = nn.Conv2d(channels, channels, kernel_size=(M, 1), stride=1, padding=(1, 0))

    def forward(self, x):
        return self.hconv(x) + self.vconv(x)
    

class CCB(nn.Module):
    def __init__(self, channels, M=3, sc=4):
        super().__init__()
        self.conv1 = CrossConv(channels, M)
        self.conv2 = CrossConv(channels, M)
        self.conv3 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.se = SqueezeExcitation(channels, squeeze_channels=sc)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.se(x)
        return x + identity
    

class GSA(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.pre_norm = ConvenientLayerNorm(channels)
        self.pre_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.qkv_conv = nn.Conv2d(channels, channels * 3, kernel_size=1, bias=False)
        self.mn_conv = nn.Linear(channels // num_heads, 2 * channels // num_heads, bias=False)
        self.gating = nn.Parameter(torch.ones(1, num_heads * 2, 1, 1))
        self.project_out = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x, condition=None):
        identity = x
        b, _, h, w = x.shape
        q, k, v = self.qkv_conv(self.pre_conv(self.pre_norm(x))).chunk(3, dim=1)

        q = q.reshape(b, self.num_heads, -1, h * w)
        k = k.reshape(b, self.num_heads, -1, h * w)
        v = v.reshape(b, self.num_heads, -1, h * w)
        q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)

        m, n = self.mn_conv(torch.matmul(q, k.transpose(-2, -1).contiguous())).chunk(2, dim=-1)  # [c, c]
        g1, g2 = self.gating.chunk(2, dim=1)

        if condition is not None:
            illu = condition.reshape(b, self.num_heads, -1, h * w)
            m = torch.matmul(m, illu)  # [c, hw]
            n = torch.matmul(n, illu)
            out = torch.softmax((g1 * m + g2 * n) * v, dim=-2).reshape(b, -1, h, w)
        else:
            out = torch.softmax(torch.matmul(g1 * m + g2 * n, v), dim=-2).reshape(b, -1, h, w)
        
        out = self.project_out(out) + identity
        return out


class FEFU(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.chn_conv = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.spa_conv = nn.Conv2d(2, 1, kernel_size=1, bias=False)
        self.project_out = nn.Conv2d(channels * 2, channels, kernel_size=1)

    def forward(self, x, y):
        chn_att = torch.sigmoid(self.chn_conv(F.adaptive_avg_pool2d(x, 1)))
        x = x * chn_att

        spa_att = torch.sigmoid(self.spa_conv(
            torch.cat([torch.max(y, dim=1, keepdim=True)[0], torch.mean(y, dim=1, keepdim=True)], dim=1)
        ))
        y = y * spa_att

        out = self.project_out(torch.cat([x, y], dim=1))
        return out
    

class DFN(nn.Module):
    def __init__(self, channels, expansion_factor=2.66):
        super().__init__()
        hidden_channels = int(channels * expansion_factor)
        self.project_in = nn.Conv2d(channels, hidden_channels * 2, kernel_size=1, bias=False)
        self.pre_norm = ConvenientLayerNorm(channels)
        self.conv = nn.Conv2d(
            hidden_channels * 2,
            hidden_channels * 2,
            kernel_size=3,
            padding=1,
            groups=hidden_channels * 2,
            bias=False,
        )
        self.project_out = nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        x1, x2 = self.conv(self.project_in(self.pre_norm(x))).chunk(2, dim=1)
        x = self.project_out(F.gelu(x1) * x2) + x
        return x
    

class CTFA(nn.Module):
    def __init__(self, channels, num_heads):
        super().__init__()
        self.ccb = CCB(channels)
        self.gsa = GSA(channels, num_heads)
        self.fefu = FEFU(channels)
        self.dfn = DFN(channels)

    def forward(self, x, condition=None):
        x1 = self.ccb(x)
        x2 = self.gsa(x, condition)
        out = self.dfn(self.fefu(x1, x2))
        return out
    

class CTFB(nn.Module):
    def __init__(self, in_channels, out_channels, num_heads):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.ctfa = CTFA(out_channels, num_heads)

    def forward(self, x, *args, **kwargs):
        return self.ctfa(self.conv(x), condition=None)


class DownSample(nn.Module):
    def __init__(self, channels):
        super(DownSample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2, kernel_size=3, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x, *args, **kwargs):
        return self.body(x)


class UpSample(nn.Module):
    def __init__(self, channels):
        super(UpSample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x, *args, **kwargs):
        return self.body(x)
    

class ConditionedSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        
    def forward(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer(x, *args, **kwargs)
        return x
    

class IlluGenerator(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, channels, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(channels, 3, kernel_size=2, stride=2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv0 = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(channels * 2, channels * 4, kernel_size=3, stride=2, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        illu = self.model(x)
        illu1 = self.lrelu(self.conv0(illu))
        illu2 = self.lrelu(self.conv1(illu1))
        illu3 = self.lrelu(self.conv2(illu2))
        return illu, [illu1, illu2, illu3]


class UNet(nn.Module):
    def __init__(self, encoders_decoders, bottleneck, depth=0):
        super().__init__()
        self.depth = depth
        outer_pair, *inner_remaining = encoders_decoders
        self.encoder, self.decoder = outer_pair
        if inner_remaining:
            self.bottleneck = UNet(inner_remaining, bottleneck, depth+1)
        else:
            self.bottleneck = bottleneck
        
    def forward(self, x, conditions):
        encoded = self.encoder(x, condition=conditions[self.depth])
        bottlenecked = self.bottleneck(encoded, conditions=conditions)
        return self.decoder(torch.cat([encoded, bottlenecked], dim=1), condition=conditions[self.depth])
    

class BrightFormer(nn.Module):
    def __init__(self, channels=48, num_heads=[1, 2, 2, 4]):
        super().__init__()
        self.embed_conv = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.illu = IlluGenerator(channels)
        self.unet = UNet([
            (ConditionedSequential(CTFA(channels, num_heads[0])), CTFB(channels * 2, channels, num_heads[0])),
            (ConditionedSequential(DownSample(channels), CTFA(channels * 2, num_heads[1])), ConditionedSequential(CTFB(channels * 4, channels * 2, num_heads[1]), UpSample(channels * 2))),
            (ConditionedSequential(DownSample(channels * 2), CTFA(channels * 4, num_heads[2])), ConditionedSequential(CTFB(channels * 8, channels * 4, num_heads[2]), UpSample(channels * 4))),
        ], ConditionedSequential(DownSample(channels * 4), CTFB(channels * 8, channels * 8, num_heads[3]), UpSample(channels * 8)))
        self.refinement = nn.Sequential(nn.Conv2d(channels, channels * 2, kernel_size=1), CTFB(channels * 2, channels, num_heads=1))
        self.output = nn.Conv2d(channels, 3, kernel_size=3, padding=1)
       
    def forward(self, x):
        identity = x
        illu, illu_feats = self.illu(x)
        x = self.embed_conv(x)
        x = self.unet(x, conditions=illu_feats)
        x = self.refinement(x)
        out = self.output(x) + identity
        return out, illu


if __name__ == '__main__':
    x = torch.randn(2, 3, 64, 64)
    model = BrightFormer()
    print(model(x)[0].shape)
