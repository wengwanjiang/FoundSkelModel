import torch, math, warnings
import numpy as np
import torch.nn as nn

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

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0., max_len: int = 200):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x) :
        x = x + self.pe[:,:x.size(1),:]
        return self.dropout(x)
    
# modality embedding
class Skeleton_Emb(nn.Module,):
    def __init__(self, t_input_size, s_input_size, hidden_size) -> None:
        super().__init__()  
        self.t_embedding = nn.Sequential(
                            nn.Linear(t_input_size, hidden_size),
                            nn.LayerNorm(hidden_size),
                            nn.ReLU(True),
                            nn.Linear(hidden_size, hidden_size),
        ) 
        self.s_embedding = nn.Sequential(
                            nn.Linear(s_input_size, hidden_size),
                            nn.LayerNorm(hidden_size),
                            nn.ReLU(True),
                            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t_src, s_src):
        t_src = self.t_embedding(t_src)
        s_src = self.s_embedding(s_src)
        return t_src, s_src
    
class DSA(nn.Module):
    def __init__(self, seqlen, dim ,gamma, gap, attn):
        super().__init__()
        self.gamma = gamma
        self.gap = gap
        self.wt1 = nn.Linear(seqlen, seqlen)
        self.wt2 = nn.Linear(seqlen, seqlen)
        self.act = nn.ReLU()
        self.drop = DropPath(0.1)
        self.attn = attn
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim*2, dim)
    def forward(self, x):
        _, MT, C = x.shape # [512, 192, 1024]
        F1 = x.permute(0, 2, 1) # [N, C, MT]

        # Dense Shift operation
        F_h = self.wt2(self.act(self.wt1(F1))) + F1 # [N, C, MT]        
        indices = torch.arange(0, MT, self.gap, device=F_h.device)
        F_h[:, :, indices] = F1[:, :, indices]
        F_h = F_h.permute(0,2,1)
        
        F_h_norm = self.norm1(F_h)
        Ftp1 = self.drop(self.attn(F_h_norm,F_h_norm,F_h_norm)[0]) + F_h
        Ftp1 = Ftp1 + self.drop(self.mlp(self.norm2(Ftp1)))

        x_norm = self.norm1(x)
        Ftp2 = self.drop(self.attn(x_norm,x_norm,x_norm)[0]) + x
        Ftp2 = Ftp2 + self.drop(self.mlp(self.norm2(Ftp2)))

        return (Ftp1 + Ftp2) * 0.5
    
class CA(nn.Module):
    def __init__(self, dim, attn, kernel_size):
        super().__init__()
        self.attn = attn
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=kernel_size)
        self.act = nn.ReLU()
        self.drop = DropPath(0.1)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, dim*2, dim)
    def forward(self, x):
        N, MT, C = x.shape # [512, 192, 1024]
        Fseq2 = x.permute(0, 2, 1) # [N, MT, C] --> [N, C, MT]      
        Ftc = (Fseq2 + self.act(self.conv1(Fseq2))).permute(0,2,1) 
        Ftc_norm = self.norm1(Ftc)
        Ftc = self.drop(self.attn(Ftc_norm,Ftc_norm,Ftc_norm)[0]) + x # [N, MT, C]
        return self.drop(self.mlp(self.norm2(Ftc)))
class DST_Layer(nn.Module):
    def __init__(self, seqlen, dim,  alpha, beta, gap, attn, kernel_size):
        super().__init__()
        self.seqlen = seqlen       
        self.CA = CA(dim, attn, kernel_size) # 引用传参  两者共享attn
        self.DSA = DSA(seqlen, dim, alpha, gap, attn) # 引用传参  两者共享attn
        self.beta = beta
        self.alpha = alpha
        #print('\nalpha, beta =', alpha, beta, '\n')
    def forward(self, x):
        return self.alpha * self.CA(x) + self.beta * self.DSA(x)
    
# Dense Spatio-Temporal Encoder
class DSTE(nn.Module):
    def __init__(self, t_input_size, s_input_size, 
                 hidden_size, num_head, num_layer, alpha, gap, kernel_size=1
                 ) -> None:
        super().__init__()
        # skeleton embedding
        self.ske_emb = Skeleton_Emb(t_input_size, s_input_size, hidden_size)
        self.d_model  = hidden_size 
        self.tpe = PositionalEncoding(hidden_size)
        self.spe = torch.nn.Parameter(torch.zeros(1, 34, hidden_size ))
        trunc_normal_(self.spe, std=.02)
        alpha, beta, gap = alpha, 1 - alpha, gap

        attn_t = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=num_head, dropout=0., batch_first=True)
        attn_s = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=num_head, dropout=0., batch_first=True)
        self.t_tr = DST_Layer(seqlen=64, dim=hidden_size, alpha=alpha, beta=beta, gap=gap, attn=attn_t, kernel_size=kernel_size)
        self.s_tr = DST_Layer(seqlen=34, dim=hidden_size, alpha=alpha, beta=beta, gap=gap, attn=attn_s, kernel_size=kernel_size)
              
        attn_t1 = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=num_head, dropout=0., batch_first=True)
        attn_s1 = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=num_head, dropout=0., batch_first=True)
        self.t_tr1 = DST_Layer(seqlen=64, dim=hidden_size, alpha=alpha, beta=beta, gap=gap, attn=attn_t1, kernel_size=kernel_size)
        self.s_tr1 = DST_Layer(seqlen=34, dim=hidden_size, alpha=alpha, beta=beta, gap=gap, attn=attn_s1, kernel_size=kernel_size)


    def forward(self, jt, js):
        t_src, s_src = self.ske_emb(jt,js)

        B, _, _ = t_src.shape
        y_t = self.t_tr(self.tpe(t_src)) 
        y_t = self.t_tr1(self.tpe(y_t)) 
        y_s = self.s_tr(s_src + self.spe.expand(B,-1,-1))
        y_s = self.s_tr1(y_s + self.spe.expand(B,-1,-1))

        return y_t, y_s

# Unified Dkeleton DENSE Representation Learnning
class USDRL(nn.Module):
    def __init__(self, t_input_size, s_input_size, 
                 hidden_size, num_head, num_layer, modality='joint', alpha=0.5, gap=4, kernel_size=1
                 ):
        super(USDRL, self).__init__()
        self.modality = modality
        self.d_model  = 2*hidden_size

        self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]
        self.backbone = DSTE(t_input_size, s_input_size, hidden_size, num_head, num_layer, alpha=alpha, gap=gap, kernel_size=kernel_size)


        # domain-specific proj
        self.i_proj = nn.Sequential(
                    nn.Linear(self.d_model, self.d_model),
                    nn.BatchNorm1d(self.d_model),
                    nn.ReLU(True),
                    nn.Linear(self.d_model, self.d_model),
                    nn.BatchNorm1d(self.d_model),
                    nn.ReLU(True),
                    nn.Linear(self.d_model, self.d_model*2),
        )
        self.s_proj = nn.Sequential(
                    nn.Linear(hidden_size, self.d_model),
                    nn.BatchNorm1d(self.d_model),
                    nn.ReLU(True),
                    nn.Linear(self.d_model, self.d_model),
                    nn.BatchNorm1d(self.d_model),
                    nn.ReLU(True),
                    nn.Linear(self.d_model, self.d_model)
        )
        self.t_proj = nn.Sequential(
                    nn.Linear(hidden_size, self.d_model),
                    nn.BatchNorm1d(self.d_model),
                    nn.ReLU(True),
                    nn.Linear(self.d_model, self.d_model),
                    nn.BatchNorm1d(self.d_model),
                    nn.ReLU(True),
                    nn.Linear(self.d_model, self.d_model)
        )
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def modality_generation(self, data_input, modality='joint'):
        N, C, T, V, M = data_input.shape
        if modality == 'joint':
            xt = data_input.permute(0, 2, 4, 3, 1)
            xt = xt.reshape(N, T, M*V*C)
            xs = data_input.permute(0, 4, 3, 2, 1)
            xs = xs.reshape(N, M*V, T*C)
        elif modality == 'bone':
            bone = torch.zeros_like(data_input)
            for v1,v2 in self.Bone:
                bone[:,:,:,v1-1,:] = data_input[:,:,:,v1-1,:] - data_input[:,:,:,v2-1,:]
                xt = bone.permute(0, 2, 4, 3, 1)
                xt = xt.reshape(N, T, M*V*C)
                xs = bone.permute(0, 4, 3, 2, 1)
                xs = xs.reshape(N, M*V, T*C)
        elif modality == 'motion':
            motion = torch.zeros_like(data_input) 
            motion[:,:,:-1,:,:] = data_input[:,:,1:,:,:] - data_input[:,:,:-1,:,:]  
            xt = motion.permute(0, 2, 4, 3, 1)
            xt = xt.reshape(N, T, M*V*C)
            xs = motion.permute(0, 4, 3, 2, 1)
            xs = xs.reshape(N, M*V, T*C)
        return xt, xs
    
    def sub_forward(self,data):
        t, s = self.modality_generation(data, self.modality)
        y_t, y_s = self.backbone(t, s)     
        y_t, y_s = y_t.amax(dim=1), y_s.amax(dim=1)
        z_t, z_s = self.t_proj(y_t), self.s_proj(y_s)
        z_i = self.i_proj(torch.cat([y_t, y_s], dim=-1))
        return z_t, z_s, z_i
    
    def forward(self, data_v1, data_v2, data_v3, data_v4):
        z_t0, z_s0, z_i0 = self.sub_forward(data_v1)
        z_t1, z_s1, z_i1 = self.sub_forward(data_v2)
        z_t2, z_s2, z_i2 = self.sub_forward(data_v3)
        z_t3, z_s3, z_i3 = self.sub_forward(data_v4)

        return [z_t0, z_t1, z_t2, z_t3], [z_s0, z_s1, z_s2, z_s3], [z_i0, z_i1, z_i2, z_i3]
        

class Downstream(nn.Module):
    def __init__(self, t_input_size, s_input_size, 
                 hidden_size, num_head, num_layer, num_class=60, modality='joint', alpha=0.5, gap=4, kernel_size=1
                 ) -> None:
        super().__init__()
        self.modality = modality
        self.d_model  = 2*hidden_size
        self.backbone = DSTE(t_input_size, s_input_size, hidden_size, num_head, num_layer, alpha=alpha, gap=gap, kernel_size=kernel_size)
        self.fc = nn.Linear(self.d_model, num_class)
    def forward(self, jt, js, bt, bs, mt, ms, knn_eval=False, detect=False):
        if self.modality == 'joint':
            y_t, y_s = self.backbone(jt, js)
        elif self.modality == 'motion':
            y_t, y_s = self.backbone(mt, ms)
        elif self.modality == 'bone':
            y_t, y_s = self.backbone(bt, bs)
        if detect == True: 
            y_s = F.adaptive_avg_pool1d(y_s.permute(0, 2, 1), 64).permute(0, 2, 1)
            y_i = torch.cat([y_t, y_s], dim=-1)
            return y_i
        y_t, y_s = y_t.amax(dim=1), y_s.amax(dim=1)
        y_i = torch.cat([y_t, y_s], dim=-1)
        if knn_eval == True:
            return y_i
        else: 
            return self.fc(y_i)
        


