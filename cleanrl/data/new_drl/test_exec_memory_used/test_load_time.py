import torch
import torchvision.models as models

alexnet = models.alexnet()

mobilenet = models.mobilenet_v2()
vgg19 = models.vgg19()
densenet201 = models.densenet201()
resnet50 = models.resnet50()

device = "cuda"
alexnet.to(device)

starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
# 同步GPU时间
starter.record()
mobilenet.to(device)
ender.record()
# 同步GPU时间
torch.cuda.synchronize()
mobilenet_load_time = starter.elapsed_time(ender) # 计算时间
print("mobilenet_load_time", mobilenet_load_time)

starter.record()
vgg19.to(device)
ender.record()
# 同步GPU时间
torch.cuda.synchronize()
vgg19_load_time = starter.elapsed_time(ender) # 计算时间
print("vgg19_load_time", vgg19_load_time)

starter.record()
densenet201.to(device)
ender.record()
# 同步GPU时间
torch.cuda.synchronize()
densenet201_load_time = starter.elapsed_time(ender) # 计算时间
print("densenet201_load_time", densenet201_load_time)

starter.record()
resnet50.to(device)
ender.record()
# 同步GPU时间
torch.cuda.synchronize()
resnet50_load_time = starter.elapsed_time(ender) # 计算时间
print("resnet50_load_time", resnet50_load_time)