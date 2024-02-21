#COPYRIGHT: HKUSTMDI LAB, THE HK UNIVERSITY OF SCIENCE & TECHNOLOGY, Guangzhou, China
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler, random_split
from torchvision import transforms
import clip
from PIL import Image
from dataset import CsvDataset
import wandb  # 导入wandb库
import sys

print("We finish loding")
import os
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '5678'
# os.environ["WANDB_MODE"]="offline"


# 初始化参数
NODE_RANK = 0  # 当前节点的rank
WORLD_SIZE = 8  # 总共的进程数
EPOCH = 40
BATCH_SIZE = 1200  #* WORLD_SIZE
VAL_BATCH_SIZE = 40
lr = 5e-5

print("EPOCH,BATCH_SIZE,VAL_BATCH_SIZE,NODE_RANK ,WORLD_SIZE,lr",EPOCH,BATCH_SIZE,VAL_BATCH_SIZE,NODE_RANK ,WORLD_SIZE,lr)

def save_checkpoint(model, optimizer, epoch, file_path):
    """保存模型和优化器的状态字典到指定路径"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, file_path)

def setup(rank, world_size):
    dist.init_process_group(
        "nccl",  # 如果你使用的是GPU，推荐使用nccl后端
        rank=rank,
        world_size=world_size
    )
    if rank==0:
        wandb.init(project="openai_clip", entity="dsa-mdil", config={"learning_rate": lr, "architecture": "ViT-B/32", "dataset": "Qulit-1M", "epochs": 40})
    print("We finished the setting up process!")

def cleanup():
    dist.destroy_process_group()
    print("We finished the cleaning up process!")

# 在train函数中添加一个evaluate函数
def evaluate(model, val_dataloader, device, top_k=(1, 5), test_tag=True, counts=10):
    """
    Usage:
    if rank == 0:
        top_k_accuracies_image, top_k_accuracies_text = evaluate(model, val_dataloader, device, top_k=(1, 5))
        print(f"Validation Top-k Image Retrieval Accuracies: {top_k_accuracies_image}")
        print(f"Validation Top-k Text Retrieval Accuracies: {top_k_accuracies_text}")
        wandb.log({'epoch': epoch, 'val_image_retrieval_acc_top1': top_k_accuracies_image[1], 'val_text_retrieval_acc_top1': top_k_accuracies_text[1]})
        wandb.log({'epoch': epoch, 'val_image_retrieval_acc_top5': top_k_accuracies_image[5], 'val_text_retrieval_acc_top5': top_k_accuracies_text[5]})
    """
    model.eval()  # Set model to evaluation mode
    image_to_text_scores = []
    text_to_image_scores = []

    with torch.no_grad():
        for i, batch in enumerate(val_dataloader):
            images, texts = batch
            images = images.to(device)
            texts = clip.tokenize(texts, truncate=True).to(device)

            # Get features for both images and text
            image_features, text_features = model(images, texts)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Calculate similarity scores
            similarity = image_features @ text_features.T
            image_to_text_scores.append(similarity)
            text_to_image_scores.append(similarity.T)
            #print("text_to_image_scores.shape, image_to_text_scores.shape: ",text_to_image_scores.shape, image_to_text_scores.shape)

            if test_tag:
                #print("i in enval process: ", i)
                if i > counts:
                    break

    # Concatenate all scores from all batches
    image_to_text_scores = torch.cat(image_to_text_scores)
    text_to_image_scores = torch.cat(text_to_image_scores)

    # Calculate Top-k accuracy for Image Retrieval and Text Retrieval
    top_k_accuracies_image = {k: 0.0 for k in top_k}
    top_k_accuracies_text = {k: 0.0 for k in top_k}

    for k in top_k:
        image_to_text_hits = image_to_text_scores.topk(k, dim=-1).indices == torch.arange(image_to_text_scores.size(0), device=device)[:, None]
        text_to_image_hits = text_to_image_scores.topk(k, dim=-1).indices == torch.arange(text_to_image_scores.size(0), device=device)[:, None]

        top_k_accuracies_image[k] = image_to_text_hits.any(dim=-1).float().mean().item()
        top_k_accuracies_text[k] = text_to_image_hits.any(dim=-1).float().mean().item()

    return top_k_accuracies_image, top_k_accuracies_text

def train(rank, world_size, test_tag = False, counts=5):
    setup(rank, world_size)

    # 设置设备
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print("The device we use are shown here: ", device)

    # 加载CLIP模型
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    print("We have loaded Vit-B/32: ")

    # 你的数据集和数据加载器
    dataset = CsvDataset("/hpc2hdd/home/wenshuozhang/wsZHANG/hanlin/hlong883_med_big_data_hlong/dataset_main/Qulit-1M/quilt_1M_lookup.csv", "image_path", "caption", sep=",", transforms=preprocess)
    train_size = int(0.99 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)

    val_dataloader = DataLoader(val_dataset, batch_size=VAL_BATCH_SIZE)
    print("Length of val_dataloader: ", len(val_dataloader))
    print("train_dataloader is prepared and its length is: ", len(dataset), train_size, val_size)
    sys.stdout.flush()

    # 封装模型
    model = DDP(model, device_ids=[rank])
    loss_img = nn.CrossEntropyLoss()
    loss_txt = nn.CrossEntropyLoss()

    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.2)

    # 训练循环
    for epoch in range(EPOCH):
        print("\n\n\n We are in epoch: ", epoch, " out of total epoch: ", EPOCH)
        sampler.set_epoch(epoch)
        for i, batch in enumerate(train_dataloader):
            #print("We are in i: ", i)
            optimizer.zero_grad()
            images, texts = batch
            images = images.to(device)
            texts = clip.tokenize(texts, truncate=True).to(device)
            if i == 0:
                print("We Finished the preprocess for images and texts")
                print("Image and Text shape: ", images.shape, texts.shape)

            logits_per_image, logits_per_text = model(images, texts)
            ground_truth = torch.arange(logits_per_image.shape[0], dtype=torch.long, device=device)
            print("The shape of Logits_pre_image, logits pre text, ground truth: ", logits_per_image.shape, logits_per_text.shape, ground_truth.shape)
            # ground_truth = torch.arange(logits_per_image.shape[0], dtype=torch.long, device=device)

            sys.stdout.flush()
            
            total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
            total_loss.backward()
            optimizer.step()

            if rank == 0 and i%1==0:  # 只有rank 0的进程记录信息
                if i%30 ==0:
                    top_k_accuracies_image, top_k_accuracies_text = evaluate(model, val_dataloader, device, top_k=(1, 5))
                    print("top_k_accuracies_image, top_k_accuracies_text: ", top_k_accuracies_image, top_k_accuracies_text)
                    wandb.log({'val_image_retrieval_acc_top1': top_k_accuracies_image[1], 'val_text_retrieval_acc_top1': top_k_accuracies_text[1]})
                    wandb.log({'val_image_retrieval_acc_top5': top_k_accuracies_image[5], 'val_text_retrieval_acc_top5': top_k_accuracies_text[5]})

                print('loss', total_loss.item())
                wandb.log({'epoch': epoch, 'loss': total_loss.item(), 'step': epoch * len(train_dataloader) + i})
            if test_tag:
                if i > counts:
                    break

        if rank == 0 and (epoch + 1) % 5 == 0:
            checkpoint_path = f"hlong883_med_big_data_hlong/ckpt/checkpoint_epoch_{epoch+1}.pth"
            save_checkpoint(model, optimizer, epoch, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1} to {checkpoint_path}")




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
