import argparse
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class image_encoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        resnet = torchvision.models.resnet50(pretrained = True)
        module = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*module)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, output_dim)
        self.output_dim = output_dim
        
    def forward(self, image):
        features = self.resnet(image)
        features = self.flatten(features)
        features = self.fc(features)
        
        return features

class text_encoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=300, hidden_dim=512, num_layers=1, bidirectional=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional)
        self.hidden_dim = hidden_dim
        self.num_directions = 2 if bidirectional else 1
    
    def forward(self, input_ids, lengths):
        embedded = self.embedding(input_ids)
        lengths = lengths.cpu()
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_out, (h_n, c_n) = self.lstm(packed)
        
        h_n = h_n.view(self.lstm.num_layers, self.num_directions, input_ids.size(0), self.hidden_dim)
        last_hidden = h_n[-1]  # (num_directions, B, hidden_dim)
        question_embedding = last_hidden.transpose(0,1).contiguous().view(input_ids.size(0), -1)
        
        return question_embedding
        

class SoftAttentionFusion(nn.Module):
    def __init__(self, img_feat_dim=1024, ques_feat_dim=1024, hidden_dim=512):
        super().__init__()
        self.img_proj = nn.Linear(img_feat_dim, hidden_dim)
        self.ques_proj = nn.Linear(ques_feat_dim, hidden_dim)
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, img_feats, ques_feat):
        """
        img_feats: (B, N, img_feat_dim) 
        ques_feat: (B, ques_feat_dim)
        """
        B, N, _ = img_feats.size()
        img_proj = self.img_proj(img_feats)               # (B, N, hidden_dim)
        ques_proj = self.ques_proj(ques_feat).unsqueeze(1).expand(-1, N, -1)  # (B, N, hidden_dim)

        joint = torch.tanh(img_proj + ques_proj)          # (B, N, hidden_dim)
        attn_logits = self.attn(joint).squeeze(-1)        # (B, N)
        attn_weights = F.softmax(attn_logits, dim=1)      # (B, N)

        attended_img = torch.sum(img_feats * attn_weights.unsqueeze(-1), dim=1)  # (B, img_feat_dim)
        fused = torch.cat([attended_img, ques_feat], dim=-1)                     # (B, img_feat_dim + ques_feat_dim)

        return fused, attn_weights

class AnswerHead(nn.Module):
    def __init__(self, input_dim, num_classes=1344):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)   
        )
    
    def forward(self, fused_features):
        logits = self.fc(fused_features)        
        probs = torch.softmax(logits, dim=-1)   
        return logits, probs


class CoralVQAModel(nn.Module):
    def __init__(self, vocab_size, num_classes=1344, 
                 img_feat_dim=1024, ques_feat_dim=1024, hidden_dim=512):
        super().__init__()
        self.image_encoder = image_encoder(output_dim=img_feat_dim)
        self.text_encoder = text_encoder(vocab_size=vocab_size, embed_dim=300, hidden_dim=512, bidirectional=True)

        self.fusion = SoftAttentionFusion(img_feat_dim=img_feat_dim,
                                          ques_feat_dim=ques_feat_dim,
                                          hidden_dim=hidden_dim)

        self.answer_head = AnswerHead(input_dim=img_feat_dim + ques_feat_dim,
                                      num_classes=num_classes)

    def forward(self, images, input_ids, lengths):
        img_feat = self.image_encoder(images)   
        img_feat = img_feat.unsqueeze(1)      

        lengths = lengths.cpu()
        ques_feat = self.text_encoder(input_ids, lengths)  

        fused, attn_weights = self.fusion(img_feat, ques_feat)  

        logits, probs = self.answer_head(fused)

        return logits, probs, attn_weights




