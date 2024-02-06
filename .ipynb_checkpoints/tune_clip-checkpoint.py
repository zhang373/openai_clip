import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision import transforms
import clip
from PIL import Image
from dataset import CsvDataset
import wandb  # 导入wandb库

print("We finish loding")
import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'



# 初始化参数
EPOCH = 40
BATCH_SIZE = 1200
NODE_RANK = 0  # 当前节点的rank
WORLD_SIZE = 8  # 总共的进程数

def setup(rank, world_size):
    dist.init_process_group(
        "nccl",  # 如果你使用的是GPU，推荐使用nccl后端
        rank=rank,
        world_size=world_size
    )
    if rank==0:
        wandb.init(project="openai_clip", entity="dsa-mdil", config={"learning_rate": 5e-6, "architecture": "ViT-B/32", "dataset": "Qulit-1M", "epochs": 40})
    print("We finished the setting up process!")

def cleanup():
    dist.destroy_process_group()
    print("We finished the cleaning up process!")

def train(rank, world_size):
    setup(rank, world_size)

    # 设置设备
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print("The device we use are shown here: ", device)

    # 加载CLIP模型
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    print("We have loaded Vit-B/32: ", model)

    # 你的数据集和数据加载器
    # ...
    dataset = CsvDataset("/hpc2hdd/home/wenshuozhang/wsZHANG/hanlin/hlong883_med_big_data_hlong/dataset_main/Qulit-1M/quilt_1M_lookup.csv", "image_path", "caption", sep=",", transforms=preprocess)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)
    print("train_dataloader is prepared and its length is: ", len(dataset))

    # 封装模型
    model = DDP(model, device_ids=[rank])
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()    

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=5e-6, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

    # 训练循环
    for epoch in range(EPOCH):
        print("We are in epoch: ", epoch, " out of total epoch: ", EPOCH)
        sampler.set_epoch(epoch)
        for i, batch in enumerate(train_dataloader):
            print("We are in epoch i: ", i)
            optimizer.zero_grad()
            images, texts = batch
            images = images.to(device)
            texts = clip.tokenize(texts, truncate=True).to(device) #
            if i==0:
                print("We Finished the preprocess for images and texts")
                print("Image and Text shape: ", images.shape, texts.shape)

            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(BATCH_SIZE, dtype=torch.long, device=device)
            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            total_loss.backward()
            optimizer.step()

            if rank == 0 and i%1==0:  # 只有rank 0的进程记录信息
                wandb.log({'epoch': epoch, 'loss': total_loss.item(), 'step': epoch * len(train_dataloader) + i})


    if rank == 0:
        wandb.finish()  # 关闭wandb
        print("wandb has been closed!")
    cleanup()

def main():
    world_size = WORLD_SIZE
    print("The total world_size is :" , world_size, "and we started to train!")
    mp.spawn(train,
             args=(world_size,),
             nprocs=world_size,
             join=True)

if __name__ == "__main__":
    main()