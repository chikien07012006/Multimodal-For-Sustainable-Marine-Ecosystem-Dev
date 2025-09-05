import os
import argparse
import torch 
import yaml
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split
from model.cnn_lstm_based.vqa_model import CoralVQAModel
from data.dataset import CoralReefVQA_Dataset, DataLoader
from matplotlib import pyplot as plt
from transformers import DistilBertTokenizer
from tqdm.autonotebook import tqdm

# =====================
# Tối ưu CUDA
# =====================
torch.backends.cudnn.benchmark = True
scaler = torch.cuda.amp.GradScaler()  # AMP scaler

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Link to config file", default="/home/24kien.dhc/Multimodal-For_SusDev/configs.yaml")
arg = parser.parse_args()

with open(arg.config, 'r') as file:
    config = yaml.safe_load(file)

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
dataset = CoralReefVQA_Dataset(split="train", data_dir=config['data_dir'], image_dir=config['image_dir'], tokenizer=tokenizer)

model = CoralVQAModel(vocab_size=tokenizer.vocab_size)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU Activated")
    torch.cuda.set_device(0)  
    print(f"Running on device: {torch.cuda.current_device()}")
    model = CoralVQAModel(vocab_size = tokenizer.vocab_size).to(device)
else:
    device = torch.device("cpu")

optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'])
cr = nn.CrossEntropyLoss()

total_size = len(dataset)
train_size = int(0.85 * total_size)
val_size = total_size - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

train_data = DataLoader(
    train_data, 
    batch_size=config['batch_size'], 
    shuffle=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)
val_data = DataLoader(
    val_data, 
    batch_size=config['batch_size'], 
    shuffle=False,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4
)

train_losses = []
val_losses = []

for epoch in range(config['epochs']):
    model.train()
    progress_bar = tqdm(train_data, colour="green")
    total_train_loss = 0
    for iter, batch in enumerate(progress_bar):
        images = batch['image'].to(device, non_blocking=True)
        question_ids = batch['question_ids'].to(device, non_blocking=True)
        answer_idx = batch['answer_idx'].to(device, non_blocking=True)

        # ========== Mixed Precision ==========
        with torch.cuda.amp.autocast():
            logits = model(images, question_ids, lengths=batch['length'].cpu())
            logits = logits[0]
            loss = cr(logits, answer_idx)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # =====================================

        total_train_loss += loss.item()
        progress_bar.set_description(
            "Epoch {}/{}. Iteration {}/{}. Train_Loss {:.3f}".format(
                epoch+1, config['epochs'], iter + 1, len(train_data), loss
            )
        )
        
    Train_avg_loss = total_train_loss / len(train_data)
    train_losses.append(Train_avg_loss)
    
    model.eval()
    total_val_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in val_data:
            images = batch['image'].to(device, non_blocking=True)
            question_ids = batch['question_ids'].to(device, non_blocking=True)
            answer_idx = batch['answer_idx'].to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                logits = model(images, question_ids, lengths=batch['length'].cpu())
                logits = logits[0]
                preds = torch.argmax(logits, dim=-1)
                loss = cr(logits, answer_idx)

            correct += (preds == answer_idx).sum().item()
            total += answer_idx.size(0)
            total_val_loss += loss.item()
    
    Val_avg_loss = total_val_loss / len(val_data)
    val_losses.append(Val_avg_loss)
    accuracy = correct / total
    
    print("Epoch {}: Accuracy: {}".format(epoch+1, accuracy))
    torch.save(model.state_dict(), os.path.join(config['save_model_dir'], f"epoch_{epoch+1}.pth"))

plt.figure(figsize=(10, 5))
plt.plot(range(1, config['epochs'] + 1), train_losses, marker='o', linestyle='-', color='b', label='Train Loss')
plt.plot(range(1, config['epochs'] + 1), val_losses, marker='o', linestyle='-', color='r', label='Val Loss')
plt.title('Training and Validation Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(config['loss_png_dir'], 'loss_plt.png'))  
plt.close()
