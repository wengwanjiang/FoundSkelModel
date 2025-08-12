import torch

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def v_ac(x):
    B, D = x.shape
    std_x = torch.sqrt(x.var(dim=0) + 1e-04)
    std_loss = torch.mean(torch.relu(1 - std_x))
    z = x - x.mean(dim=0)
    cov_z = (z.T @ z) / (B - 1) 
    cov_loss = off_diagonal(cov_z).pow(2).sum() / D
    return 5 * std_loss + cov_loss

def xcorr_loss(z1, z2):
    # cross-correlation matrix
    N = z1.size(0)
    D = z2.size(1)

    z1_norm = (z1 - z1.mean(0)) / z1.std(0)
    z2_norm = (z2 - z2.mean(0)) / z2.std(0)
    c = torch.matmul(z1_norm.t(), z2_norm) / N # D D 

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()

    cross_correlation = on_diag + 0.005 * off_diag # 0.005
    return cross_correlation


def similarity(z_list, criterion): 
    assert len(z_list) == 4
    z0, z1, z2, z3 = z_list
    center = (z0 + z1 + z2) / 3

    intra = criterion(z0, z3) + criterion(z2, z3) + criterion(z1, z3)\

    inter = criterion(z0, center) + criterion(z1, center) + criterion(z2, center)
    return intra + inter


def cal_xc(z_list):
    assert len(z_list) == 4
    a, b, c, d = z_list
    return xcorr_loss(a,b) + xcorr_loss(a,c) + xcorr_loss(a,d) + \
            xcorr_loss(b,c) + xcorr_loss(b,d) + xcorr_loss(c,d)