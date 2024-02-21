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

"""
Usage:
if rank == 0:
    top_k_accuracies_image, top_k_accuracies_text = evaluate(model, val_dataloader, device, top_k=(1, 5))
    print(f"Validation Top-k Image Retrieval Accuracies: {top_k_accuracies_image}")
    print(f"Validation Top-k Text Retrieval Accuracies: {top_k_accuracies_text}")
    wandb.log({'epoch': epoch, 'val_image_retrieval_acc_top1': top_k_accuracies_image[1], 'val_text_retrieval_acc_top1': top_k_accuracies_text[1]})
    wandb.log({'epoch': epoch, 'val_image_retrieval_acc_top5': top_k_accuracies_image[5], 'val_text_retrieval_acc_top5': top_k_accuracies_text[5]})
"""
counts=3
rank=0
top_k=(1, 5)
# 设置设备
device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
print("The device we use are shown here: ", device)

# 加载CLIP模型
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
print("We have loaded Vit-B/32: ")
model.eval()  # Set model to evaluation mode
image_to_text_scores = []
text_to_image_scores = []

with torch.no_grad():
    for i in range(60):
        images, texts = torch.rand(6, 3, 224, 224), ["I am great I think", "I am great I think","I am great I think", "I am great I think", "I am great I think","I am great I think"]
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
        print("text_to_image_scores.shape, image_to_text_scores.shape: ",len(text_to_image_scores), len(image_to_text_scores))

        if True:
            print("i in enval process: ", i)
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


