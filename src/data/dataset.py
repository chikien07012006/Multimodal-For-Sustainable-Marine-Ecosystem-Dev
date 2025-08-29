import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from transformers import AutoImageProcessor, AutoTokenizer

class CoralReefVQA_Dataset(Dataset):
    def __init__(self, split, data_dir, image_dir):
        super().__init__()
        json_file_path = os.path.join(data_dir, f"CoralVQA_{split}.jsonl")
        
        self.data = []
        with open(json_file_path, 'r') as file:
            for line in file:
                self.data.append(json.loads(line))
                
        self.image_dir = image_dir
        self.image_processor = AutoImageProcessor.from_pretrained("microsoft/swin-base-simmim-window6-192")
        self.tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
                
    def __len__(self):
        return (len(self.data))
    
    def __getitem__(self, index):
        item = self.data[index]
        
        image_path = os.path.join(self.image_dir, item['image'])
        image = Image.open(image_path).convert("RGB")
        
        image = self.image_processor(images=image, return_tensors="pt")['pixel_values'].squeeze(0)
        
        question = item['conversations'][0]['value'][14:]
        answer = item['conversations'][1]['value']
        
        q_input = self.tokenizer(
            question,
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True
        )
        a_input = self.tokenizer(
            answer,
            return_tensors="pt",
            padding="max_length",
            max_length=32,
            truncation=True
        )

        return {
            'image': image,  # Tensor [3, 224, 224]
            'question_ids': q_input['input_ids'].squeeze(0),  # Tensor [64]
            'question_mask': q_input['attention_mask'].squeeze(0),  # Tensor [64]
            'answer_ids': a_input['input_ids'].squeeze(0),  # Tensor [32]
            'answer_mask': a_input['attention_mask'].squeeze(0)  # Tensor [32]
        }
        




# Data augmentation

# from torchvision import transforms
# self.transform = transforms.Compose([
#     transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(brightness=0.2, contrast=0.2)
# ])

# Áp dụng trong __getitem__ trước self.processor.


# Early stopping: Theo dõi validation loss, dừng nếu không cải thiện sau 3 epochs.