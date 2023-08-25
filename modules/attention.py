import torch
import torch.nn as nn
from .transformer import PositionalEncoding

class Attention(nn.Module):
    def __init__(self, in_channels=512, max_length=25, n_feature=256):
        super().__init__()
        self.max_length = max_length

        self.f0_embedding = nn.Embedding(max_length, in_channels)
        self.w0 = nn.Linear(max_length, n_feature)
        self.wv = nn.Linear(in_channels, in_channels)
        self.we = nn.Linear(in_channels, max_length)

        self.active = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, enc_output):
        enc_output = enc_output.permute(0, 2, 3, 1).flatten(1, 2)
        reading_order = torch.arange(self.max_length, dtype=torch.long, device=enc_output.device)
        reading_order = reading_order.unsqueeze(0).expand(enc_output.size(0), -1)  # (S,) -> (B, S)
        reading_order_embed = self.f0_embedding(reading_order)  # b,25,512

        t = self.w0(reading_order_embed.permute(0, 2, 1))  # b,512,256
        t = self.active(t.permute(0, 2, 1) + self.wv(enc_output))  # b,256,512

        attn = self.we(t)  # b,256,25
        attn = self.softmax(attn.permute(0, 2, 1))  # b,25,256
        g_output = torch.bmm(attn, enc_output)  # b,25,512
        return g_output, attn.view(*attn.shape[:2], 8, 32)


def encoder_layer(in_c, out_c, k=3, s=2, p=1):
    return nn.Sequential(nn.Conv2d(in_c, out_c, k, s, p),
                         nn.BatchNorm2d(out_c),
                         nn.ReLU(True))

def decoder_layer(in_c, out_c, k=3, s=1, p=1, mode='nearest', scale_factor=None, size=None):
    align_corners = None if mode=='nearest' else True
    return nn.Sequential(nn.Upsample(size=size, scale_factor=scale_factor, 
                                     mode=mode, align_corners=align_corners),
                         nn.Conv2d(in_c, out_c, k, s, p),
                         nn.BatchNorm2d(out_c),
                         nn.ReLU(True))


class PositionAttention(nn.Module):
    def __init__(self, max_length, in_channels=512, num_channels=64, 
                 h=8, w=32, mode='nearest', **kwargs):
        super().__init__()
        self.max_length = max_length
        self.k_encoder = nn.Sequential(
            encoder_layer(in_channels, num_channels, s=(1, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2)),
            encoder_layer(num_channels, num_channels, s=(2, 2))
        )
        self.k_decoder = nn.Sequential(
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, num_channels, scale_factor=2, mode=mode),
            decoder_layer(num_channels, in_channels, size=(h, w), mode=mode)
        )

        self.pos_encoder = PositionalEncoding(in_channels, dropout=0, max_len=max_length)
        self.project = nn.Linear(in_channels, in_channels)
    
    #注释以radical的attention为例，radical最大长度为33+1=34
    def forward(self, x):
        N, E, H, W = x.size()
        k, v = x, x  # (N, E, H, W) = [44, 512, 8, 32]

        # calculate key vector
        features = []
        for i in range(0, len(self.k_encoder)):
            k = self.k_encoder[i](k)
            features.append(k)
        for i in range(0, len(self.k_decoder) - 1):
            k = self.k_decoder[i](k)
            k = k + features[len(self.k_decoder) - 2 - i]
        k = self.k_decoder[-1](k)

        # calculate query vector
        # TODO q=f(q,k)
        zeros = x.new_zeros((self.max_length, N, E))  # (T, N, E)   #T为最大长度，N为一次传入的图片数，E为embedding后向量维度
        q = self.pos_encoder(zeros)  # (T, N, E)      # 计算位置编码
        q = q.permute(1, 0, 2)  # (N, T, E)
        q = self.project(q)  # (N, T, E)              # 乘上Q矩阵
        
        #print("q:",q.size())         # q:[44, 34, 512]
        #print("k:",k.size())         # k:[44, 512, 8, 32]
        
        # calculate attention
        attn_scores = torch.bmm(q, k.flatten(2, 3))  # k:(N, T, (H*W)) = [44, 512, 256]       # Q*K(T)
        #print("attn_scores:",attn_scores.size())     # attn_scores:[44, 34, 256]
        
        attn_scores = attn_scores / (E ** 0.5)                                                # Q*K(T)/√E
        attn_scores = torch.softmax(attn_scores, dim=-1)                                      # softmax(Q*K(T)/√E) 在最后一个维度上做softmax

        v = v.permute(0, 2, 3, 1).view(N, -1, E)   # (N, (H*W), E)   [44, 512, 8, 32]->[44, 8, 32, 512]->[44, 256, 512]
        #print("v:",v.size())                       # v:[44, 256, 512]
        
        attn_vecs = torch.bmm(attn_scores, v)  # (N, T, E)  [44, 34, 256]*[44, 256, 512]=[44, 34, 512]    # softmax(Q*K(T)/√E) * V
        
        return attn_vecs, attn_scores.view(N, -1, H, W)
