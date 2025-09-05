import torch
import torch.nn as nn
from transformers import SwinModel, BertModel, GPT2LMHeadModel

class CoralVQAModel(nn.Module):
    def __init__(self, hidden_size = 768, bert_size = 30522, device=None):
        super().__init__()
        self.hidden = hidden_size
        self.image_encoder = SwinModel.from_pretrained("microsoft/swin-base-simmim-window6-192")
        self.text_encoder = BertModel.from_pretrained("google-bert/bert-base-uncased")
        self.decoder = GPT2LMHeadModel.from_pretrained("gpt2").transformer
        self.cross_attn = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8, dropout=0.1)
        self.lm_head = nn.Linear(hidden_size, bert_size) #anwsers được tokenized = bert => có vocab_size=30522, nên cần 
        # này để ánh xạ lại câu trl từ head gpt2 (vocab_size = 50257) để đồng bộ hoá số logits, để tính hàm loss
        self.img_linear = nn.Linear(self.image_encoder.config.hidden_size, hidden_size)
        
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        if device is not None:
            self.to(device)
        
    def forward(self, images, question_ids, question_mask, answer_ids=None):
        img_feats = self.image_encoder(images).last_hidden_state  
        img_feats = self.img_linear(img_feats)
        
        text_feats = self.text_encoder(input_ids=question_ids, attention_mask=question_mask).last_hidden_state  # [batch, 64, 768]
        
        fused_feats, _ = self.cross_attn(
            text_feats.transpose(0, 1),  # Query: [64, batch, 768]
            img_feats.transpose(0, 1),  # Key: [49, batch, 768]
            img_feats.transpose(0, 1)   # Value
        )
        fused_feats = fused_feats.transpose(0, 1)  # [batch, 64, 768]
        
        # Decoder
        if answer_ids is not None:  
            seq_len = answer_ids.shape[1]
            fused_feats = fused_feats[:, :seq_len, :]  # [batch, seq_len, hidden]
            fused_feats = fused_feats.reshape(fused_feats.shape).contiguous()  # Đảm bảo contiguous
            output = self.decoder(inputs_embeds=fused_feats).last_hidden_state
            logits = self.lm_head(output)  # [batch, seq_len, num_classes]
            return logits
        else:  # Inference
            # xử lý trong infer.py
            return fused_feats