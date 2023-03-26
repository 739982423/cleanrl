import torch
import torchvision.models as models

# Pytorch 内部有自己的缓存管理系统，能够加速显存分配。
# 使用 torch.cuda.memory_allocated() 可以看到当前Tensor占用的显存
# 使用 torch.cuda.memory_reserved() 可以看到pytorch总共占用的显存
# 使用 torch.cuda.empty_cache() 清空未使用的缓存，但是已经使用的是不能释放的


def get_memory_allocated():
    return torch.cuda.memory_allocated(device="cuda") / 1024 / 1024

def get_memory_reserved():
    return torch.cuda.memory_reserved(device="cuda") / 1024 / 1024

# flag = "allocated"
flag = "reserved"

device = "cuda"

if flag == "allocated":
    origin_memory = get_memory_allocated()
else:
    origin_memory = get_memory_reserved()

print("origin memory:", origin_memory)

model = models.vgg19().to(device)

if flag == "allocated":
    m1 = get_memory_allocated()
else:
    m1 = get_memory_reserved()

print("after loading model:", m1)


for b in [256]:
    torch.cuda.empty_cache()
    print("------------------------------------")
    dummy_input = torch.rand((b,3,224,224)).to(device)
    print("input size = {}*3*224*224 = {} MB".format(b, b*3*224*224/1024/1024))

    if flag == "allocated":
        m2 = get_memory_allocated()
    else:
        m2 = get_memory_reserved()

    print("after loading input tensor:", m2)
    
    with torch.no_grad():
        i = 1
        while(i <= 10000):
            _ = model(dummy_input)
            i += 1
            if i % 25 == 0:
                if flag == "allocated":
                    m4 = get_memory_allocated()
                else:
                    m4 = get_memory_reserved()

                print("looping memory:", m4)
    
    if flag == "allocated":
        m5 = get_memory_allocated()
    else:
        m5 = get_memory_reserved()
    print("exec end:", m5)
    