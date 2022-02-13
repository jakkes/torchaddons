import torch

v = torch.tensor([
    [[1.0, 0.0],
    [0.0, 1.0]],

    [[100, 99],
    [99, 100]],

    [[1.0, 0.1],
    [0.1, 1.0]]
])

d = torch.distributions.MultivariateNormal(
    torch.randn(3, 2),
    v
)

print(d.sample())