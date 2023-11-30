import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch import einsum
import numpy as np
import torch_dct as dct

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# Layer Normalization
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)

# The Frequency Domain Learning
class SpecDCT(nn.Module):
    def __init__(self,dim,heads,dim_head,h,w):
        super().__init__()
        self.dim=dim
        self.heads=heads
        self.dim_head=dim_head
        self.h=h
        self.w=w

        # temp=self.dim//28

        self.to_q=nn.Linear(dim,dim_head*heads,bias=False)
        self.to_k=nn.Linear(dim,dim_head*heads,bias=False)
        self.to_v=nn.Linear(dim,dim_head*heads,bias=False)
        self.rescale=nn.Parameter(torch.ones(heads,1,1))
        # position embedding
        # self.pos_emb=nn.Parameter(torch.Tensor(1,heads,dim_head,dim_head))
        # trunc_normal_(self.pos_emb)
        
        self.proj=nn.Linear(dim_head*heads,dim,bias=True)

        self.high_freq_conv1=nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU()
        )
        self.high_freq_conv2=nn.Sequential(
            nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
            GELU()
        )
        # Learnable Gating Filter
        self.coef_emb=nn.Parameter(torch.ones(h,w))
        self.Eone=torch.ones(h,w).cuda().float()

    def forward(self,x_input):

        bs,height,width,nc=x_input.shape
        kernel=8
        kernel_num1=height//kernel
        kernel_num2=width//kernel
        x_in=x_input.permute(0,3,1,2)
        
        x_dct=dct.dct_2d(x_in)

        # Spectral-wise self-Attention of Frequency
        x_in1=rearrange(x_dct,'b c (hh m0) (ww m1) -> b (hh ww) c m0 m1',hh=kernel_num1,ww=kernel_num2,m0=kernel,m1=kernel)
        x_dct_new=rearrange(x_in1,'b n c m0 m1 -> b n (m0 m1) c')
        x_tri=x_dct_new
        q_dct=self.to_q(x_tri)
        k_dct=self.to_k(x_tri)
        v_dct=self.to_v(x_tri)
        q_dct,k_dct,v_dct=map(lambda t: rearrange(t,'b n mm (h d) -> b n h mm d',h=self.heads),(q_dct,k_dct,v_dct))
        q_dct,k_dct,v_dct=map(lambda t: t.transpose(-2,-1),(q_dct,k_dct,v_dct))
        q_dct=F.normalize(q_dct,dim=-1,p=2)
        k_dct=F.normalize(k_dct,dim=-1,p=2)
        attn=(q_dct@k_dct.transpose(-2,-1))
        attn=attn*self.rescale
        # position embedding
        # attn=attn+self.pos_emb
        attn=attn.softmax(dim=-1)
        x0=attn@v_dct
        x1=rearrange(x0,'b n h d mm -> b n mm (h d)',h=self.heads,d=self.dim_head)
        x2=self.proj(x1)
        x3=rearrange(x2,'b n (m0 m1) c -> b n c m0 m1',m0=kernel,m1=kernel)
        x_low_attn=rearrange(x3,'b (hh ww) c m0 m1 -> b c (hh m0) (ww m1)',hh=kernel_num1,ww=kernel_num2,m0=kernel,m1=kernel)

        # spectral-spatial interaction of frequency (SIF)
        x_rbtri_highfreq=x_dct
        x_conv_highfreq=self.high_freq_conv1(x_rbtri_highfreq)+x_rbtri_highfreq
        x_high_conv=self.high_freq_conv2(x_conv_highfreq)+x_conv_highfreq

        # Learnable Gating Filter
        expand_coef=self.coef_emb.expand([bs,nc,height,width])
        Eone=self.Eone.expand([bs,nc,height,width])
        coef_high=Eone-expand_coef

        # Frequency Level Gating
        x_out=expand_coef*x_low_attn+coef_high*x_high_conv
        x_out=x_out+x_dct
        x_out=dct.idct_2d(x_out).permute(0,2,3,1)
        
        return x_out

# The Space Domain Learning
class localAttn(nn.Module):
    def __init__(self,dim,heads,dim_head,h,w,window_size):
        super().__init__()
        self.dim=dim
        self.heads=heads
        self.dim_head=dim_head
        self.h=h
        self.w=w
        self.window_size=window_size

        # temp=self.dim//28

        self.to_q=nn.Linear(dim,dim_head*heads,bias=False)
        self.to_kv=nn.Linear(dim,dim_head*heads*2,bias=False)
        # self.to_v=nn.Linear(dim,dim_head*heads,bias=False)
        # self.rescale=nn.Parameter(torch.ones(heads,1,1))
        self.rescale=dim_head**-0.5
        # position embedding
        seq_l=window_size[0]*window_size[1]
        self.pos_emb=nn.Parameter(torch.Tensor(1,heads,seq_l,seq_l))
        trunc_normal_(self.pos_emb)
        
        self.proj=nn.Linear(dim_head*heads,dim,bias=True)

    def forward(self,x_input):

        bs,h,w,nc=x_input.shape
        # x_in=x_input.permute(0,3,1,2)
        
        x_inp = rearrange(x_input, 'b (h b0) (w b1) c -> (b h w) (b0 b1) c', b0=self.window_size[0], b1=self.window_size[1])
        q = self.to_q(x_inp)
        k, v = self.to_kv(x_inp).chunk(2, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))
        q *= self.rescale
        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim = sim + self.pos_emb
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.proj(out)
        out = rearrange(out, '(b h w) (b0 b1) c -> b (h b0) (w b1) c', h=h // self.window_size[0], w=w // self.window_size[1],
                            b0=self.window_size[0],b1=self.window_size[1])
        
        return out

# Mixing Domains Learning Block
class HS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            window_size=(8, 8),
            dim_head=28,
            heads=8,
            channel=28,
            height=256,
            width=320,
            only_local_branch=False
    ):
        super().__init__()

        self.dim = dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.window_size = window_size
        self.channel=channel
        self.height=height
        self.width=width
        self.only_local_branch = only_local_branch

        inner_dim = dim_head * heads

        # The Frequency Domain Learning
        self.spec_attn=SpecDCT(dim=self.dim,heads=self.heads,dim_head=dim_head,h=height//heads,w=width//heads)
        # The Space Domain Learning
        self.local_attn=localAttn(dim=self.dim,heads=self.heads,dim_head=dim_head,h=height,w=width,window_size=window_size)
        self.fusion=nn.Conv2d(self.dim*2,self.dim,1,1,0,bias=True)

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x.shape
        w_size = self.window_size
        assert h % w_size[0] == 0 and w % w_size[1] == 0, 'fmap dimensions must be divisible by the window size'

        # The Frequency Domain Learning
        out_fd=self.spec_attn(x)
        # The Space Domain Learning
        out_local=self.local_attn(x)

        # Mixing Domains
        spec_local=torch.cat([out_fd,out_local],dim=-1)
        out=self.fusion(spec_local.permute(0,3,1,2)).permute(0,2,3,1)

        return out

# Correlation-driven Mixing Domains Transformer
class HSAB(nn.Module):
    def __init__(
            self,
            dim,
            window_size=(8, 8),
            dim_head=64,
            heads=8,
            channel=28,
            height=256,
            width=320,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                PreNorm(dim, HS_MSA(dim=dim, window_size=window_size, dim_head=dim_head, heads=heads,channel=channel,height=height,width=width, only_local_branch=(heads==1))),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1) # b,h,w,c
        for (attn, ff) in self.blocks:
            x = attn(x) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out

# Feed-Forward Network
class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)

# The U-shaped Prior Module
class HST(nn.Module):
    def __init__(self, in_dim=28, out_dim=28, dim=28, num_blocks=[1,1,1]):
        super(HST, self).__init__()
        self.dim = dim
        self.scales = len(num_blocks)

        # Input projection
        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_scale = dim
        for i in range(self.scales-1):
            self.encoder_layers.append(nn.ModuleList([
                HSAB(dim=dim_scale, num_blocks=num_blocks[i], dim_head=dim, heads=dim_scale // dim),
                nn.Conv2d(dim_scale, dim_scale * 2, 4, 2, 1, bias=False),
            ]))
            dim_scale *= 2

        # Bottleneck
        self.bottleneck = HSAB(dim=dim_scale, dim_head=dim, heads=dim_scale // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(self.scales-1):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_scale, dim_scale // 2, stride=2, kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_scale, dim_scale // 2, 1, 1, bias=False),
                HSAB(dim=dim_scale // 2, num_blocks=num_blocks[self.scales - 2 - i], dim_head=dim,
                     heads=(dim_scale // 2) // dim),
            ]))
            dim_scale //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        #### activation function
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """

        b, c, h_inp, w_inp = x.shape
        hb, wb = 16, 16
        pad_h = (hb - h_inp % hb) % hb
        pad_w = (wb - w_inp % wb) % wb
        x = F.pad(x, [0, pad_w, 0, pad_h], mode='reflect')

        # Embedding
        fea = self.embedding(x)
        x = x[:,:28,:,:]

        # Encoder
        fea_encoder = []
        for (HSAB, FeaDownSample) in self.encoder_layers:
            fea = HSAB(fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)

        # Bottleneck
        fea = self.bottleneck(fea)

        # Decoder
        for i, (FeaUpSample, Fution, HSAB) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(torch.cat([fea, fea_encoder[self.scales-2-i]], dim=1))
            fea = HSAB(fea)

        # Mapping
        out = self.mapping(fea) + x
        return out[:, :, :h_inp, :w_inp]

def A(x,Phi):
    temp = x*Phi
    y = torch.sum(temp,1)
    return y

def At(y,Phi):
    temp = torch.unsqueeze(y, 1).repeat(1,Phi.shape[1],1,1)
    x = temp*Phi
    return x

def shift_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=step*i, dims=2)
    return inputs

def shift_back_3d(inputs,step=2):
    [bs, nC, row, col] = inputs.shape
    for i in range(nC):
        inputs[:,i,:,:] = torch.roll(inputs[:,i,:,:], shifts=(-1)*step*i, dims=2)
    return inputs

# The Iteration Parameter Estimator
class HyPaNet(nn.Module):
    def __init__(self, in_nc=29, out_nc=8, channel=64):
        super(HyPaNet, self).__init__()
        self.fution = nn.Conv2d(in_nc, channel, 1, 1, 0, bias=True)
        self.down_sample = nn.Conv2d(channel, channel, 3, 2, 1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp_9stg = nn.Sequential(
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, out_nc, 1, padding=0, bias=True),
                nn.Softplus())
        self.relu = nn.ReLU(inplace=True)
        self.out_nc = out_nc

    def forward(self, x):
        x = self.down_sample(self.relu(self.fution(x)))
        x = self.avg_pool(x)
        x = self.mlp_9stg(x) + 1e-6
        return x[:,:self.out_nc//2,:,:], x[:,self.out_nc//2:,:,:]

# Deep Frequency Unfolding Framework
class DAUHST(nn.Module):

    def __init__(self, num_iterations=9):
        super(DAUHST, self).__init__()
        self.para = HyPaNet(in_nc=28, out_nc=num_iterations*2)
        self.fution = nn.Conv2d(56, 28, 1, padding=0, bias=True)
        self.num_iterations = num_iterations
        self.denoiser=HST(in_dim=29, out_dim=28, dim=28, num_blocks=[1,1,1])
    def initial(self, y, Phi):
        """
        :param y: [b,256,310]
        :param Phi: [b,28,256,310]
        :return: temp: [b,28,256,310]; alpha: [b, num_iterations]; beta: [b, num_iterations]
        """
        nC, step = 28, 2
        y = y / nC * 2
        bs,row,col = y.shape
        y_shift = torch.zeros(bs, nC, row, col).cuda().float()
        for i in range(nC):
            y_shift[:, i, :, step * i:step * i + col - (nC - 1) * step] = y[:, :, step * i:step * i + col - (nC - 1) * step]
        z = self.fution(torch.cat([y_shift, Phi], dim=1))
        alpha, beta = self.para(self.fution(torch.cat([y_shift, Phi], dim=1)))
        return z, alpha, beta

    def forward(self, y, input_mask=None):
        """
        :param y: [b,256,310]
        :param Phi: [b,28,256,310]
        :param Phi_PhiT: [b,256,310]
        :return: z_crop: [b,28,256,256]
        """
        Phi, Phi_s = input_mask
        
        # The IPE
        z, alphas, betas = self.initial(y, Phi)
        for i in range(self.num_iterations):
            alpha, beta = alphas[:,i,:,:], betas[:,i:i+1,:,:]
            Phi_z = A(z, Phi)
            
            # The Data Module
            x = z + At(torch.div(y-Phi_z,alpha+Phi_s), Phi)
            x = shift_back_3d(x)
            beta_repeat = beta.repeat(1,1,x.shape[2], x.shape[3])
            
            # The Prior Module
            z = self.denoiser(torch.cat([x, beta_repeat],dim=1))
            if i<self.num_iterations-1:
                z = shift_3d(z)
        return z[:, :, :, 0:256]
