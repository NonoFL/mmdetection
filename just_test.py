import torch
cls = torch.tensor([[0.1,0.2,0.3,0.4,0.5,0.6],[0.7,0.6,0.5,0.4,0.3,0.2,0.1]], requires_grad=True)
G_cls = torch.tensor([[0,0,0,0,1,0], [0,1,0,0,0,0]])
G_loc = torch.tensor([True])