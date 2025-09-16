import os
from collections import defaultdict
import torch
import torch.nn as nn
import torchvision
from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from auto_LiRPA.utils import Flatten

class problem_vertification(nn.Module):
    def __init__(self):
        super(problem_vertification, self).__init__()
        self.layer1 = nn.Linear(2, 2, bias=True)
        self.layer2 = nn.Linear(2, 2, bias=True)
        self.layer3 = nn.Linear(2, 1, bias=False)

        with torch.no_grad():
            self.layer1.weight.data = torch.tensor([[1.0, -1.0], [2.0, -2.0]])
            self.layer1.bias.data = torch.tensor([1.0, 1.0])
            self.layer2.weight.data = torch.tensor([[1.0, -1.0], [2.0, -2.0]])
            self.layer2.bias.data = torch.tensor([2.0, 2.0])
            self.layer3.weight.data = torch.tensor([[-1.0, 1.0]])
    
    def forward(self, x):
        z1 = self.layer1(x)
        h1 = torch.relu(z1)
        z2 = self.layer2(h1)
        h2 = torch.relu(z2)
        output = self.layer3(h2 + z1)
        return output

if __name__ == '__main__':
    model = problem_vertification().eval()

    lirpa = BoundedModule(model, torch.empty(1,2))

    x = torch.tensor([[0.0, 0.0]])
    ptb = PerturbationLpNorm(norm=float('inf'), eps=1)
    x = BoundedTensor(x, ptb)

    lb, ub = lirpa.compute_bounds(x=(x,), method='CROWN')
    print("CROWN:")
    print(lb, ub)

    lb, ub = lirpa.compute_bounds(x=(x,), method='IBP')
    print("IBP:")
    print(lb, ub)
