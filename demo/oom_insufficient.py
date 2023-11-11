import torch
# Check CUDA memory before allocation
print('Before allocation',torch. cuda. max_memory_allocated())
# Allocate a large tensor to cause OOM
x = torch.randn(1,30*1024*1024*1024, device='cuda')
# Check CUDA memory after allocation
print('After allocation',torch. cuda. max_memory_allocated())
