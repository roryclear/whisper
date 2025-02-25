from tinygrad import Tensor
import torch
torch.manual_seed(42)
x = torch.rand(1)
print(x)

x_tiny = Tensor(x.numpy())
print(x_tiny.numpy())