"""Torch models are kept separate from the workload builder."""

import torch


class Matmul(torch.nn.Module):
    def forward(self, a, b):
        return a @ b


class BatchMatmul(torch.nn.Module):
    def forward(self, a, b):
        return torch.bmm(a, b)


class MatmulChain(torch.nn.Module):
    def forward(self, a, b, c):
        return (a @ b) @ c
