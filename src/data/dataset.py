import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from collections import defaultdict

class CoralReefVQA_Dataset(Dataset):
    def __init__(self, split, data_dir, image_dir, tokenizer, max_q_len=32, transform=None):
        
        super().__init__()
        json_file_path = os.path.join(data_dir, f"CoralVQA_{split}.jsonl")
        
        self.data = []
        with open(json_file_path, 'r') as file:
            for line in file:
                self.data.append(json.loads(line))

        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_q_len = max_q_len
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                     std=[0.229, 0.224, 0.225])
            ])
        else:
            self.transform = transform

        self.answer2idx = {}
        if split == "train":
            answers = set()
            for item in self.data:
                ans = item['conversations'][1]['value']
                answers.add(ans)
            self.answer2idx = {ans: idx for idx, ans in enumerate(sorted(list(answers)))}
            self.num_classes = len(self.answer2idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        image_path = os.path.join(self.image_dir, item['image'])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)  

        question = item['conversations'][0]['value'][14:] 
        answer = item['conversations'][1]['value']

        tokens = self.tokenizer(question)
        input_ids = tokens['input_ids'][:self.max_q_len]
        length = len(input_ids)

        if length < self.max_q_len:
            pad_len = self.max_q_len - length
            input_ids = input_ids + [0] * pad_len  
        
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        length = torch.tensor(length, dtype=torch.long)

        if answer in self.answer2idx:
            answer_idx = torch.tensor(self.answer2idx[answer], dtype=torch.long)
        else:
            answer_idx = torch.tensor(-1, dtype=torch.long)

        return {
            'image': image,                  
            'question_ids': input_ids,       
            'length': length,                
            'answer_idx': answer_idx         
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