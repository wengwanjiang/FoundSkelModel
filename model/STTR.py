import torch
import torch.nn.functional as F
import torch.nn as nn
import warnings, math

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


# spatio-temporal transformer encoder
class STTR(nn.Module):
    def __init__(self, t_input_size, s_input_size, 
                 hidden_size, num_head, num_layer
                 ) -> None:
        super().__init__()
        # skeleton embedding
        self.ske_emb = Skeleton_Emb(t_input_size, s_input_size, hidden_size)
        # skeleton encoder
        self.d_model  = hidden_size 
        self.tpe = PositionalEncoding(hidden_size)
        self.spe = torch.nn.Parameter(torch.zeros(1, 34, hidden_size ))
        trunc_normal_(self.spe, std=.02)
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        t_layer = TransformerEncoderLayer(self.d_model , num_head, self.d_model , batch_first = True, dropout=0.) 
        self.t_tr = TransformerEncoder(t_layer, num_layer)
        s_layer = TransformerEncoderLayer(self.d_model , num_head, self.d_model , batch_first = True, dropout=0.)
        self.s_tr = TransformerEncoder(s_layer, num_layer)

    def forward(self, jt, js):
        # embedding
        t_src, s_src = self.ske_emb(jt,js)

        B, _, _ = t_src.shape
        y_t = self.t_tr(self.tpe(t_src)) 
        y_s = self.s_tr(s_src + self.spe.expand(B,-1,-1))

        return y_t, y_s

# Unified Dkeleton DENSE Representation Learnning
class USDRL(nn.Module):
    def __init__(self, t_input_size, s_input_size, 
                 hidden_size, num_head, num_layer, modality='joint', alpha=0.5, kernel_size=1, gap=4
                 ):
        super(USDRL, self).__init__()
        self.modality = modality
        self.d_model  = 2*hidden_size

        self.Bone = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]
        self.backbone = STTR(t_input_size, s_input_size, hidden_size, num_head, num_layer)

        print('Use: ', self.modality, 'to train the model')
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
        #self.apply(self._init_weights)
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
        self.backbone = STTR(t_input_size, s_input_size, hidden_size, num_head, num_layer)
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
            return self.fc(y_i)
        y_t, y_s = y_t.amax(dim=1), y_s.amax(dim=1)
        y_i = torch.cat([y_t, y_s], dim=-1)
        if knn_eval == True:
            return y_i
        else: 
            return self.fc(y_i)
        


