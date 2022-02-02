"""
Autograd
https://pytorch.org/blog/overview-of-pytorch-autograd-engine/
Pytorch computation Graph
Pytorch Internals
http://blog.ezyang.com/2019/05/pytorch-internals/
"""

# simple auto grad backward example
import torch

# ref
# https://pytorch.org/blog/overview-of-pytorch-autograd-engine/
x = torch.tensor([0.5, 0.75], requires_grad=True)
y = torch.log(x[0] * x[1]) * torch.sin(x[1])
y.backward(torch.tensor(1.0))  # the initial z=w, dz/dw=1.0

print(x.grad)

