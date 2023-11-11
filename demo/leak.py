import torch
import sys
sys.path.append("..")

#x = torch.randn(100, 100, requires_grad=False).cuda()
x = torch.randn(8000, 8000, requires_grad=True).cuda()

print("before accumulate gradients")

# Create an empty list to accumulate gradients
accumulated_gradients = []

# Accumulate gradients without using them
for i in range(1000000):
    y = x * 2
    z = y.mean()


    # Uncommenting the following line fixes the memory leak
    #z.backward()

    accumulated_gradients.append(z)
