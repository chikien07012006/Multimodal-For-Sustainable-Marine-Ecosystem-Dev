import os
import json
from tqdm import tqdm
from functools import partial
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from omegaconf import OmegaConf

from open_flamingo import create_model_and_transforms 
from open_flamingo.train.any_res_data_utils import process_images

# 路径设置
model_ckpt = "/root/autodl-tmp/LAVIS/finetune-xgenmmv1-phi3_4k_instruct-CoralVQA/checkpoint_14999.pt"
input_jsonl = "/root/autodl-tmp/CoralVQA/bleaching_area.jsonl"
output_jsonl = "/root/autodl-tmp/LAVIS/results/bleaching_area.jsonl"
image_root = "/root/autodl-tmp/CoralVQA/bleaching"

# 模型配置
cfg = dict(
    model_family='xgenmm_v1',
    lm_path='microsoft/Phi-3-mini-4k-instruct',
    vision_encoder_path='google/siglip-so400m-patch14-384',
    vision_encoder_pretrained='google',
    num_vision_tokens=128,
    image_aspect_ratio='anyres',
    anyres_patch_sampling=True,
    anyres_grids=[(1,2), (2,1), (2,2), (3,1), (1,3)],
    ckpt_pth=model_ckpt,
)
cfg = OmegaConf.create(cfg)

# 初始化模型
additional_kwargs = {
    "num_vision_tokens": cfg.num_vision_tokens,
    "image_aspect_ratio": cfg.image_aspect_ratio,
    "anyres_patch_sampling": cfg.anyres_patch_sampling,
}
model, image_processor, tokenizer = create_model_and_transforms(
    clip_vision_encoder_path=cfg.vision_encoder_path,
    clip_vision_encoder_pretrained=cfg.vision_encoder_pretrained,
    lang_model_path=cfg.lm_path,
    tokenizer_path=cfg.lm_path,
    model_family=cfg.model_family,
    **additional_kwargs
)
ckpt = torch.load(cfg.ckpt_pth)["model_state_dict"]
model.load_state_dict(ckpt, strict=True)
torch.cuda.empty_cache()
model = model.eval().cuda()

# 设置 anyres grid
base_img_size = model.base_img_size
model.anyres_grids = [[base_img_size * m, base_img_size * n] for (m, n) in cfg.anyres_grids]

# 图像处理器
image_proc = partial(process_images, image_processor=image_processor, model_cfg=cfg)

# Prompt 模板
def apply_prompt_template(prompt, cfg):
    if 'Phi-3' in cfg.lm_path:
        return (
            '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
            "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
            f'<|user|>\n{prompt}<|end|>\n<|assistant|>\n'
        )
    else:
        raise NotImplementedError

kwargs_default = dict(do_sample=False, temperature=0, max_new_tokens=1024, top_p=None, num_beams=1)

# 读取已有预测结果
existing_ids = set()
if os.path.exists(output_jsonl):
    with open(output_jsonl, 'r') as f:
        for line in f:
            try:
                existing_ids.add(json.loads(line)['question_id'])
            except:
                continue

# 读取输入数据
with open(input_jsonl, 'r') as f:
    input_data = [json.loads(line) for line in f if json.loads(line)['question_id'] not in existing_ids]

# 推理函数（线程任务）
def process_item(item):
    question_id = item['question_id']
    image_name = item['image']
    text = item['text']
    image_path = os.path.join(image_root, image_name)

    if not os.path.exists(image_path):
        return None  # 图像缺失

    try:
        image = Image.open(image_path).convert('RGB')
        vision_x = [image_proc([image])]
        vision_x = [vision_x]
        image_size = [[image.size]]

        prompt = apply_prompt_template(text, cfg)
        lang_x = tokenizer([prompt], return_tensors="pt")

        with torch.no_grad():
            generated_ids = model.generate(
                vision_x=vision_x,
                lang_x=lang_x['input_ids'].to(torch.device('cuda:0')),
                image_size=image_size,
                attention_mask=lang_x['attention_mask'].to(torch.device('cuda:0')),
                **kwargs_default
            )

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        predicted = generated_text.split('<|end|>')[0] if 'Phi-3' in cfg.lm_path else generated_text

        return {
            "question_id": question_id,
            "image": image_name,
            "text": text,
            "predicted": predicted.strip()
        }
    except Exception as e:
        print(f"Error processing question_id {question_id}: {e}")
        return None

# 多线程执行并写入
max_workers = 4  # 可根据 GPU 占用和 I/O 性能调整
with open(output_jsonl, 'a') as f_out:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_item, item) for item in input_data]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating"):
            result = future.result()
            if result:
                f_out.write(json.dumps(result, ensure_ascii=False) + '\n')
                f_out.flush()
