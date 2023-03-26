import torch
import torchvision.models as models

device = "cuda"

resnet50 = models.resnet50().to(device)

input_tensor = torch.rand((1,3,224,224)).to(device)
# print(input_tensor)
print(input_tensor.shape)
while(1):
    _ = resnet50(input_tensor)
