import torch
import torch.nn as nn
import torch.nn.functional as F
from sparsemax import Sparsemax


class TabNet(nn.Module):
    def __init__(
        self,
        n_features,
        n_targets,
        na=16,
        nd=16,
        n_steps=1,
        n_blocks=1,
        n_shared_blocks=1,
        shared_mask=False,
        relax=1,
        ghost_size=0,
        sparsemax=True,
        sparse_loss=False,
    ):
        super().__init__()
        self.na = na
        self.n_steps = n_steps
        self.relax = relax
        self.sparse_loss = sparse_loss
        self.bn = nn.BatchNorm1d(n_features)
        self.relu = nn.ReLU()
        size = na + nd
        h = nn.Linear(na, n_features, bias=False) if shared_mask else None
        lins = [nn.Linear(n_features, size)]
        for block in range(n_shared_blocks):
            lins.append(nn.Linear(size, size))
        n_unshared = n_shared_blocks - n_blocks
        self.first_step = TabFeatures(lins, n_unshared, size, ghost_size)
        self.steps = nn.ModuleList(
            [
                TabStep(h, n_features, na, nd, lins, n_unshared, ghost_size, sparsemax)
                for step in range(n_steps)
            ]
        )
        self.head = nn.Linear(nd, n_targets)

    def forward(self, x):
        x = self.bn(x)
        xout = 0
        p = torch.ones(x.shape)
        xa = self.first_step(x)[:, : self.na]
        sparse_loss = 0
        for step in self.steps:
            xa, xd, m = step(x, xa, p)
            p = p * (self.relax - m)
            xout += self.relu(xd)
            sparse_loss += ((-m * torch.log(m + 1e-08)).sum()) / x.shape[0]
        if self.sparse_loss:
            return self.head(xout), sparse_loss / self.n_steps
        else:
            return self.head(xout)


class GhostNorm(nn.Module):
    def __init__(self, size, ghost_size):
        super().__init__()
        self.size = size
        self.bn = nn.BatchNorm1d(size)
        self.ghost_size = ghost_size

    def forward(self, x):
        chunks = torch.chunk(x, x.size(0) // self.ghost_size, 0)
        res = [self.bn(y) for y in chunks]
        return torch.cat(res, axis=0)


class TabMask(nn.Module):
    def __init__(self, n_features, h=None, na=None, ghost_size=0, sparsemax=True):
        super().__init__()
        if h == None:
            h = nn.Linear(na, n_features, bias=False)
        self.h = h
        self.bn = (
            GhostNorm(n_features, ghost_size)
            if ghost_size
            else nn.BatchNorm1d(n_features)
        )
        self.sm = Sparsemax() if sparsemax else nn.Softmax()

    def forward(self, xa, p):
        a = self.bn(self.h(xa))
        m = self.sm(p * a)
        return m


class TabFeatures(nn.Module):
    def __init__(self, lins, n_unshared, size, ghost_size):
        super().__init__()
        self.first_block = FeatBlock(lins[0], size, ghost_size)
        self.blocks = nn.ModuleList()
        for lin in lins[1:]:
            self.blocks.append(FeatBlock(lin, size, ghost_size))
        for n in range(n_unshared):
            lin = nn.Linear(size, size)
            self.blocks.append(FeatBlock(lin, size, ghost_size))

    def forward(self, x):
        x = self.first_block(x)
        for block in self.blocks:
            o = block(x)
            x = (o + x) * (0.5 ** 0.5)
        return x


class FeatBlock(nn.Module):
    def __init__(self, lin, size, ghost_size):
        super().__init__()
        self.lin = lin
        self.bn = GhostNorm(size, ghost_size) if ghost_size else nn.BatchNorm1d(size)

    def forward(self, x):
        x = self.bn(self.lin(x))
        x = torch.cat((x, x), axis=1)
        return F.glu(x)


class TabStep(nn.Module):
    def __init__(self, h, n_features, na, nd, lins, n_unshared, ghost_size, sparsemax):
        super().__init__()
        self.mask = TabMask(n_features, h, na, ghost_size, sparsemax)
        self.features = TabFeatures(lins, n_unshared, na + nd, ghost_size)
        self.na = na

    def forward(self, x, xa, p):
        m = self.mask(xa, p)
        mx = m * x
        o = self.features(mx)
        xa = o[:, : self.na]
        xd = o[:, self.na :]
        return xa, xd, m
