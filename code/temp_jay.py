import torch
import numpy as np

a = np.array([10, 2, 7, 31, 34, 3, 1])
x = torch.from_numpy(a)


b = np.array([10, 13, 12, 5, 6, 8, 10])
y = torch.from_numpy(b)

print(x)
print(y)
print('yes')

x_, sorted_indices = x.sort(0, descending=True)
y_ = y[sorted_indices]

print(x_)
print(sorted_indices)
print(y_)
print('yes')

sorted_indices, sorted_indices_ = sorted_indices.sort(0, descending=False)
a_ = y_[sorted_indices_]

print(sorted_indices)
print(sorted_indices_)
print(a_)