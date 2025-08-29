import os
import argparse
import torch 
import yaml
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split
from model.vqa_model import CoralVQAModel
from data.dataset import CoralReefVQA_Dataset, DataLoader
from matplotlib import pyplot as plt
from transformers import AutoTokenizer
from tqdm.autonotebook import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--config", help="Link to config file", default="/home/24kien.dhc/Multimodal-For_SusDev/configs.yaml")
arg = parser.parse_args()

with open(arg.config, 'r') as file:
    config = yaml.safe_load(file)
    
import torch
import torch.nn as nn

if torch.cuda.is_available():
    # Đặt CUDA_VISIBLE_DEVICES để chỉ dùng GPU 6 và 7
    device = torch.device("cuda")
    print("GPU Activated")
    # Sử dụng GPU 6 và 7 (thay vì tất cả GPU)
    torch.cuda.set_device(0)  # GPU 6 sẽ là cuda:0, GPU 7 là cuda:1 khi chỉ định 6,7
    print(f"Running on device: {torch.cuda.current_device()}")
    model = CoralVQAModel(device=device).to(device)
    # Sử dụng DataParallel với GPU 6 và 7
    if torch.cuda.device_count() >= 2:
        print('1')
        model = nn.DataParallel(model, device_ids=[0, 1])  # GPU 6, 7
else:
    device = torch.device("cpu")

optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad == True], lr = config['lr'])
cr = nn.CrossEntropyLoss(ignore_index=0)

dataset = CoralReefVQA_Dataset(split="train", data_dir=config['data_dir'], image_dir=config['image_dir'])

total_size = len(dataset)
train_size = int(0.85 * total_size)
val_size = total_size - train_size
train_data, val_data = random_split(dataset, [train_size, val_size])

train_data = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
val_data = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False)

train_losses = []
val_losses = []

for epoch in range(config['epochs']):
    model.train()
    progress_bar = tqdm(train_data, colour = "green")
    total_train_loss = 0
    for iter, batch in enumerate(progress_bar):
        images = batch['image'].to(device)
        question_ids = batch['question_ids'].to(device)
        question_mask = batch['question_mask'].to(device)
        answer_ids = batch['answer_ids'].to(device)
        
        logits = model(images, question_ids, question_mask, answer_ids)
        logits_flat = logits.reshape(-1, logits.size(-1))
        answer_ids_flat = answer_ids.reshape(-1)
        loss = cr(logits_flat, answer_ids_flat)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        progress_bar.set_description("Epoch {}/{}. Iteration {}/{}. Train_Loss {:.3f}".format(epoch+1, config['epochs'], iter + 1, len(train_data), loss))
        
    Train_avg_loss = total_train_loss / len(train_data)
    train_losses.append(Train_avg_loss)
    
    model.eval()
    total_val_loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in val_data:
            images = batch['image'].to(device)
            question_ids = batch['question_ids'].to(device)
            question_mask = batch['question_mask'].to(device)
            answer_ids = batch['answer_ids'].to(device)
            
            logits = model(images, question_ids, question_mask, answer_ids)
            preds = torch.argmax(logits, dim=-1)
            loss = cr(logits.view(-1, logits.size(-1)), answer_ids.view(-1))
            
            tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
            
            for pred, true in zip(preds, answer_ids):
                pred_text = tokenizer.decode(pred, skip_special_tokens=True)
                true_text = tokenizer.decode(true, skip_special_tokens=True)
                if pred_text.strip() == true_text.strip():
                    correct += 1
                total += 1
            
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
plt.savefig(os.path.join(config['loss_png_dir', 'loss_plt.png']))  
plt.close()