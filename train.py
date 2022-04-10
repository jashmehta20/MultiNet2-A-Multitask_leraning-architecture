import torch

from multinet import Multinet

model = Multinet()

batch = torch.rand(5, 3, 1248, 384)

model, batch = model.to('cpu'), batch.to('cpu')

x1, x2 = model(batch)

print(x1.shape)
print(x2.shape)