import os
from collections import defaultdict
import torch
import torch.nn as nn
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import Flatten

from model import SimpleNNRelu, SimpleNNHardTanh

def verify_network(model, x, eps, method):
    model.eval()
    lirpa = BoundedModule(model, torch.empty_like(x))
    ptb = PerturbationLpNorm(norm=float('inf'), eps=eps)
    x = BoundedTensor(x, ptb)
    lb, ub = lirpa.compute_bounds(x=(x,), method=method)
    return lb, ub

def print_bounds_crown_style(lb, ub, title="results"):
    print(f"\n{title}:")
    for i in range(lb.shape[1]):
        print('f_{j}(x_{i}): {l:8.4f} <= f_{j}(x_{i}+delta) <= {u:8.4f}'.format(
            j=i, i=0, l=lb[0, i].item(), u=ub[0, i].item()))


if __name__ == '__main__':
    x_test, label = torch.load('data1.pth')
    batch_size = x_test.size(0)
    x = x_test.reshape(batch_size, -1)

    relu_model = SimpleNNRelu()
    relu_model.load_state_dict(torch.load('models/relu_model.pth'))
    relu_model.eval()

    relu_output = relu_model(x)
    print(f"relu_output: {relu_output}")
    relu_lb, relu_ub = verify_network(relu_model, x, 0.01, 'CROWN-Optimized')

    hardtanh_model = SimpleNNHardTanh()
    hardtanh_model.load_state_dict(torch.load('models/hardtanh_model.pth'))
    hardtanh_model.eval()

    hardtanh_output = hardtanh_model(x)
    print(f"hardtanh_output: {hardtanh_output}")
    hardtanh_lb, hardtanh_ub = verify_network(hardtanh_model, x, 0.01, 'CROWN-Optimized')

    print_bounds_crown_style(relu_lb, relu_ub, title="relu")
    print_bounds_crown_style(hardtanh_lb, hardtanh_ub, title="hardtanh")
