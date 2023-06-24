import torch
from torchvision import models as models
from multiprocessing import Process, Barrier, Lock, Value
import argparse
import csv
import os
import time

def run(id, barrier, batch, gpu_resource, exec_ratio, loop_times, file_name):
    # print("子进程id:", os.getpid())

    global lock1, lock2, lock3, Counter, TotalModels, Prepared

    id_model_hash = {
        0 : "resnet50",
        1 : "vgg19",
        2 : "densenet201",
        3 : "mobilenet",
    }
    device = torch.device('cuda')
    model = None
    if id == 0:
        model = models.resnet50().to(device)
    elif id == 1:
        model = models.vgg19().to(device)
    elif id == 2:
        model = models.densenet201().to(device)
    elif id == 3:
        model = models.mobilenet.mobilenet_v2().to(device)

    dummy_input = torch.randn(batch,3,224,224, dtype=torch.float).to(device)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    torch.cuda.empty_cache()
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    with lock3:
        Prepared.value += 1

    print("{} prepared! waiting...".format(id_model_hash[id]))
    barrier.wait()
    total_time = 0
    real_exec = 0
    with torch.no_grad():
        for _ in range(loop_times):
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)

            total_time += curr_time
            real_exec += 1
            if Counter.value >= 1:
                break
    print("{}: gpu = {}, bs = {}, 执行了{}次, total latency = {}, avg latency = {}".format(id_model_hash[id], gpu_resource, batch, real_exec, total_time, total_time / real_exec))
    with lock1:
        Counter.value += 1
    with lock2:
        with open(file_name, mode="a", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            # title = ["model name", "gpu resource", "batchsize", "total latency", "real exec times", "avg latency"]
            real_exec_times = total_time / real_exec
            throughput = 1000 / real_exec_times * batch
            ratio_throughput = exec_ratio * throughput
            row = [id_model_hash[id], gpu_resource, batch, throughput, ratio_throughput]
            writer.writerow(row)
            # # 如果当前是最后执行完成的模型，则写入一个空行，以分隔每次运行的结果
            # if Counter.value == TotalModels.value:
            #     writer.writerow([])
        f.close()

def initialize_mps():
    os.system("echo quit | sudo nvidia-cuda-mps-control")
    time.sleep(0.5)
    os.system("sudo nvidia-cuda-mps-control -d")

def set_mps_gpu(gpu_percent):
    os.system("echo set_default_active_thread_percentage {} | sudo nvidia-cuda-mps-control".format(gpu_percent))
    time.sleep(0.5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m1", "--model1", help="需要被测试的model1", type=int)
    parser.add_argument("-m2", "--model2", help="需要被测试的model2", type=int)
    parser.add_argument("-b1", "--batch1", help="传入的batch1", type=int)
    parser.add_argument("-b2", "--batch2", help="传入的batch2", type=int)
    parser.add_argument("-g1", "--gpu1", help="分配的GPU资源1", type=int)
    parser.add_argument("-g2", "--gpu2", help="分配的GPU资源2", type=int)
    parser.add_argument("-id", "--gpuidx", help="分配到的GPU编号", type=int)
    parser.add_argument("-gr1", "--g1ratio", help="GPU1的可用时间比例", type=float)
    parser.add_argument("-gr2", "--g2ratio", help="GPU2的可用时间比例", type=float)
    parser.add_argument("-t", "--time", help="时序流中的时刻", type=int)

    args = parser.parse_args() 
    id_model_hash = {
        0 : "resnet50",
        1 : "vgg19",
        2 : "densenet201",
        3 : "mobilenet",
    }
    
    # 这两个变量与同时执行的模型数量有关
    barrier = Barrier(2)        
    TotalModels = Value('i', 2) # 表示当前同时执行的模型数量

    lock1 = Lock()  # 用来隔离更新Counter的数值，让所有子进程判断是否有已执行完的模型
    lock2 = Lock()  # 用来隔离向临时文件写入子进程的运行结果
    lock3 = Lock()  # 子进程准备完毕时改变全局变量prepared时需要的锁

    Counter = Value('i', 0)
    Prepared = Value('i', 0)

    model_exec_times = 100

    # 创建一个文件夹保存当前时刻两个GPU的结果（每个GPU的执行结果用一个csv保存）
    
    csv_name = "gpu{}.csv".format(args.gpuidx)

    # 临时文件名称，用来存储3个子进程的运行结果，在主进程中读取每次运行的结果，进行排序后再存入最终的csv文件
    tmp_csv_path = "./res/origin_data/{}/gpu{}.csv".format(args.time, args.gpuidx)

    ratio = 1
    if args.gpuidx == 0:
        ratio = args.g1ratio
    elif args.gpuidx == 1:
        ratio = args.g2ratio
    # 准备好共存的进程
    p1 = Process(target=run, args=(args.model1, barrier, args.batch1, args.gpu1, ratio, model_exec_times, tmp_csv_path))
    p2 = Process(target=run, args=(args.model2, barrier, args.batch2, args.gpu2, ratio, model_exec_times, tmp_csv_path))

    # 初始化MPS
    initialize_mps()

    # 设置第一个模型的GPU分配量
    set_mps_gpu(args.gpu1)
    
    p1.start()

    # 等待p1准备完毕
    while(1):
        with lock3:
            if Prepared.value == 1:
                break
    time.sleep(0.5)

    # 设置第二个模型的GPU分配量
    server_id = os.popen("echo get_server_list | nvidia-cuda-mps-control").readlines()[0].strip('\n')
    act_thread1 = os.popen("echo get_active_thread_percentage {} | sudo nvidia-cuda-mps-control".format(server_id)).readlines()[0].strip('\n')
    gpu_set_cmd = "echo set_active_thread_percentage {} {} | sudo nvidia-cuda-mps-control".format(server_id, args.gpu2)
    os.system(gpu_set_cmd)
    act_thread2 = os.popen("echo get_active_thread_percentage {} | sudo nvidia-cuda-mps-control".format(server_id)).readlines()[0].strip('\n')


    p2.start()
    
   
    # 等待执行结束
    p1.join()
    p2.join()