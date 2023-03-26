import torch
import torchvision.models as models

resnet50 = models.resnet50(pretrained=True)
densenet201 = models.densenet201(pretrained=True)
vgg19 = models.vgg19(pretrained=True)
mobilenet =  models.mobilenet_v2(pretrained=True)
torch.save(resnet50, "./resnet50.pt")
torch.save(densenet201, "./densenet201.pt")
torch.save(vgg19, "./vgg19.pt")
torch.save(mobilenet, "./mobilenet_v2.pt")


