import torch
import pdb
#x = torch.randn(100, 100, requires_grad=False).cuda()
# y = torch.empty(8000)

# Doing so will not release GPU memory
# x = torch.randn(8000, 8000, requires_grad=True)
while(1):
    input('Waiting for debugger')
    x = torch.randn(8000, 8000, requires_grad=True)
# y = x.cuda()
# z = y
# print(id(x),id(y), id(z))
# print(x.type(), y.type(), z.type())
