import torch
from transformers import BertTokenizer
from src.model.vqa_model import CoralVQAModel
from src.data.dataset import CoralVQADataset

def generate_answer(model, image, question, tokenizer, device, max_length=32):
    model.eval()
    image = image.to(device)
    q_input = tokenizer(question, return_tensors="pt", padding="max_length", max_length=64, truncation=True)
    question_ids = q_input['input_ids'].to(device)
    question_mask = q_input['attention_mask'].to(device)
    
    with torch.no_grad():
        fused_feats = model(image, question_ids, question_mask)  # [batch, seq_len, hidden]
    
    # Greedy decoding
    generated = [tokenizer.cls_token_id]
    for _ in range(max_length):
        input_ids = torch.tensor([generated]).to(device)
        logits = model.decoder(input_ids).h[-1][:, -1, :]  # Last token
        logits = model.lm_head(logits)
        next_token = torch.argmax(logits, dim=-1).item()
        generated.append(next_token)
        if next_token == tokenizer.sep_token_id:
            break
    
    return tokenizer.decode(generated, skip_special_tokens=True)

# Test
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CoralVQAModel().to(device)
    model.load_state_dict(torch.load('models/checkpoints/epoch_10.pth'))
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    dataset = CoralVQADataset(split='test', data_dir='data/raw')
    sample = dataset[0]
    answer = generate_answer(model, sample['image'].unsqueeze(0), 
                           tokenizer.decode(sample['question_ids'], skip_special_tokens=True),
                           tokenizer, device)
    print(f"Question: {tokenizer.decode(sample['question_ids'], skip_special_tokens=True)}")
    print(f"Answer: {answer}")