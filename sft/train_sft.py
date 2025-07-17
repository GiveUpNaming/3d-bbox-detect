import os
import json
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image

import transformers
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2VLProcessor,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
)
# from peft import LoraConfig, get_peft_model

# --- 1. 定义命令行参数 ---
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="Qwen/Qwen2.5-7B-VL-Instruct")

@dataclass
class DataArguments:
    data_path: str = field(default="qwen_sft_data.json", metadata={"help": "Path to the preprocessed training data."})
    image_folder: Optional[str] = field(default=None, metadata={"help": "Path to the folder where the images are."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=2048, metadata={"help": "Maximum sequence length."})
    use_lora: bool = field(default=True, metadata={"help": "Whether to use LoRA."})


# --- 2. 自定义数据集 ---
class SFTDataset(Dataset):
    def __init__(self, data_path: str, processor: Qwen2VLProcessor, image_folder: str = None):
        print("Loading data...")
        self.list_data_dict = json.load(open(data_path, "r"))
        self.processor = processor
        self.image_folder = image_folder
        print(f"Data loaded. Found {len(self.list_data_dict)} samples.")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.list_data_dict[i]
        
        # 1. Load the image
        image_file = item['image']
        if self.image_folder is not None:
            image_file = os.path.join(self.image_folder, image_file)
        
        try:
            image = Image.open(image_file).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file not found at {image_file}. Skipping to next item.")
            return self.__getitem__((i + 1) % len(self))
            
        # 2. Correctly format the conversation for Qwen2-VL
        # The model needs to know that an image is part of the user's prompt.
        # We transform your data format into the one the processor expects.
        # User: {"from": "user", "value": "..."} -> AI: {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "..."}]}
        
        user_turn = item['conversations'][0]
        assistant_turn = item['conversations'][1]

        messages = [
            {
                "role": user_turn['from'],
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": user_turn['value']}
                ]
            },
            {
                "role": assistant_turn['from'],
                "content": assistant_turn['value']
            }
        ]

        # 3. Use the processor to get the full model inputs
        # The processor takes care of tokenizing text, processing the image, and
        # critically, creating the 'image_grid_thw'.
        # We process text and images together.
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        model_inputs = self.processor(
            text=text, 
            images=[image], 
            return_tensors="pt"
        )
        
        # 4. PRESERVE ALL PROCESSOR OUTPUTS
        # The key fix: We squeeze the batch dimension but keep the entire dictionary
        # that the processor returned. This ensures 'image_grid_thw' is not lost.
        final_inputs = {key: val.squeeze(0) for key, val in model_inputs.items()}

        # 5. Create labels robustly for Supervised Fine-Tuning (SFT)
        # We mask out the user's prompt so the model only learns to predict the assistant's response.
        
        # To do this, we find where the assistant's response begins.
        # We apply the template to ONLY the user's turn to get the length of the prompt.
        prompt_only_messages = [messages[0]]
        prompt_text = self.processor.apply_chat_template(
            prompt_only_messages, 
            tokenize=False, 
            add_generation_prompt=True # IMPORTANT: This adds the <|im_start|>assistant\n tokens
        )
        prompt_ids = self.processor.tokenizer(prompt_text).input_ids
        prompt_length = len(prompt_ids)

        # Create labels and mask the prompt part
        labels = final_inputs['input_ids'].clone()
        labels[:prompt_length] = -100
        
        # Also mask any padding tokens in the labels
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Add the final labels to our dictionary
        final_inputs['labels'] = labels
        
        return final_inputs



# --- 3. 训练主函数 ---
def train():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 加载模型和Processor
    # 使用 bfloat16 加速，device_map="auto" 自动分配显存
    # use_flash_attention_2=True 在支持的硬件上能极大提速
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        # device_map="auto",
        use_flash_attention_2=True,
        trust_remote_code=True,
    )
    processor = Qwen2VLProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    
    # 如果使用LoRA
    # if training_args.use_lora:
    #     print("Using LoRA for training...")
    #     lora_config = LoraConfig(
    #         r=8,
    #         lora_alpha=16,
    #         target_modules=["c_attn", "attn_c_proj", "w1", "w2"],
    #         lora_dropout=0.05,
    #         bias="none",
    #         task_type="CAUSAL_LM",
    #     )
    #     model = get_peft_model(model, lora_config)
    #     model.print_trainable_parameters()

    # 加载数据集
    dataset = SFTDataset(data_path=data_args.data_path, processor=processor, image_folder=data_args.image_folder)
    # sample = dataset[0]
    # print(sample)
    # print(data_args)

    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        # collate_fn 需要处理padding，但我们的dataset __getitem__ 已经返回了处理好的tensor
        # HuggingFace Trainer 的默认 collate_fn 就能很好地处理这种情况
    )


    # 开始训练
    trainer.train()

    # 保存模型
    # 如果使用LoRA，只会保存adapter部分，非常小
    # 如果是全量微调，会保存整个模型
    trainer.save_model(training_args.output_dir)
    print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    train()