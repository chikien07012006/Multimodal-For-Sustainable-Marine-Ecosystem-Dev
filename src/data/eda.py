import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import shutil

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

train_data = load_jsonl(r"C:\Users\Administrator\Desktop\Multimodal-For_SusDev\data\CoralVQA_train.jsonl")
test_data = load_jsonl(r"C:\Users\Administrator\Desktop\Multimodal-For_SusDev\data\CoralVQA_train.jsonl")

train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

def extract_answers(conv_list):
    answers = []
    for c in conv_list:
        if c["from"] == "gpt":
            answers.append(c["value"])
    return answers

train_df["answers"] = train_df["conversations"].apply(extract_answers)
train_df["answer_len"] = train_df["answers"].apply(
    lambda ans: [len(a.split()) for a in ans] if isinstance(ans, list) else []
)

all_answer_lengths = [l for sublist in train_df["answer_len"].tolist() for l in sublist]

print("Answer length stats:")
print(pd.Series(all_answer_lengths).describe())
unique_answers = set([answer for sublist in train_df["answers"].tolist() for answer in sublist])
print(f"Number of unique answers: {len(unique_answers)}")

plt.hist(all_answer_lengths, bins=40, color="seagreen", edgecolor="black")
plt.title("Distribution of Answer Lengths (Train set)")
plt.xlabel("Number of tokens")
plt.ylabel("Frequency")
output_path = os.path.join(r"C:\Users\Administrator\Desktop\Multimodal-For_SusDev\src\data", "answer_length_distribution.png")
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"Histogram saved to: {output_path}")

unique_lengths = set(all_answer_lengths)
    
example_answers = {}
answer_counts = {}  

for length in unique_lengths:
    count = 0  
    for answers in train_df["answers"]:
        for answer in answers:
            if len(answer.split()) == length:
                count += 1  
                if length not in example_answers:  
                    example_answers[length] = answer
    answer_counts[length] = count  


print("Example answers and counts for unique lengths:")
for length in sorted(example_answers.keys()):
    print(f"Length {length}: Example Answer: {example_answers[length]} | Count: {answer_counts[length]}")

