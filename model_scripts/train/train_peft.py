#region Script Setup
#---Import Libraries
from unsloth import is_bf16_supported
from transformers import BitsAndBytesConfig, Idefics3ForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
import torch
from torchmetrics import F1Score
import numpy as np
import regex
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import wandb
import os
import yaml, argparse

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#---Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to YAML Config')
args, unkown = parser.parse_known_args()

#---Load YAML Config
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

model_name = config['model_name']
try:
    save_repo = config['save_repo']
    save = True
except KeyError:
    print("Not saving")
    save = False
try:
    run_name = config['run_name']
except KeyError:
    pass
try:
    job_type = config['job_type']
except:
    job_type = ""
trc = config['trc']
try:
    r_config = config['r']
    learning_rate_config = config['learning_rate']
    weight_decay_config = config['weight_decay']
except KeyError:
    pass

#---Weights & Biases Info
os.environ["WANDB_LOG_MODEL"] = "end"
os.environ["WANDB_PROJECT"] = "AgriPath-VLM-Sweep"
wandb.login(key=os.getenv("WANDB_API_KEY"))
#endregion

#region Data Collator
class SmolCollator:
    def __init__(self, processor):
        self.processor = processor
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]
        # self.anchor_text = " Class:"
        self.anchor_ids = [9519, 9531, 42]
    
    def __call__(self, examples):
        texts = []
        images = []

        # Prepare Text and Images
        for example in examples:
            image = example['image']
            question = "Identify the crop and disease in the image."
            answer = f"Class: {example['crop']}\nDisease: {example['disease']}"
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are an expert pathologist and need to identify the crop and disease present in an image. If it is a healthy crop, classify it as healthy"}]},
                {"role": "user", "content": [{"type": "text", "text": question}, {"type": "image"}]},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            ]
            
            text = self.processor.tokenizer.apply_chat_template(
                messages, 
                tokenize=False,
                add_generation_prompt=False
            )
            texts.append(text)
            images.append([image])
        
        # Tokenize the batch
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        # Create the labels by cloning input_ids
        labels = batch["input_ids"].clone()

        # Masking logic
        IGNORE_INDEX = -100

        # Iterate over the batch (x8)
        for i in range(labels.shape[0]):
            input_ids_list = labels[i].tolist()
            start_index = -1

            len_anchor = len(self.anchor_ids)
            for k in range(len(input_ids_list) - len_anchor+1):
                if input_ids_list[k:k+len_anchor] == self.anchor_ids: # Check if slice matches anchor
                    start_index = k
                    break
            
            # Crash Guard (Preventing NaN)
            if start_index == -1:
                print("\n[FATAL ERROR] Could not find Answer Anchor in token sequence!")
                print(f"Could not find anchor IDs: {self.anchor_ids}")
                print(f"Input IDs: {input_ids_list}")
                print(f"Full Text: {texts[i]}")
                raise ValueError("Collator failed: Tokenizer did not produce expected 'Class:' tokens. Training aborted to prevent NaN.")
            
            labels[i, :start_index] = IGNORE_INDEX
        
        labels[labels == self.processor.tokenizer.pad_token_id] = IGNORE_INDEX
        labels[labels == self.image_token_id] = IGNORE_INDEX
        batch["labels"] = labels

        # #DEBUG BLOCK
        # print("=== SmolCollator Debug ===")
        # print(f"Batch keys: {list(batch.keys())}")
        # print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        # print(f"Batch labels shape: {batch['labels'].shape}")
        # print(f"First labels row: {batch['labels'][0] if batch['labels'].shape[0] > 0 else 'EMPTY'}")
        # print("==========================")

        return batch
#endregion


def main():
    run = wandb.init(
        config=config,
        job_type=job_type
    )
    #region Model Call and Setup
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_in_8bit=False,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = Idefics3ForConditionalGeneration.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
    )

    for name, module in model.named_modules():
        if "norm" in name.lower():  # Targets LayerNorm, RMSNorm, etc.
            module = module.to(torch.float32)
    
    processor = AutoProcessor.from_pretrained(model_name)

    if job_type == "train_frozen_vision":
        targets=[
            "q_proj", "k_proj", "v_proj", "o_proj", # LLM Attention
            "gate_proj", "up_proj", "down_proj",   # LLM MLP
            # "fc1", "fc2",                         # Vision Encoder MLP
            # "attn.qkv", "attn.proj"              # Vision Encoder Attention
        ]
    else:
        targets=[
            "q_proj", "k_proj", "v_proj", "o_proj", # LLM Attention
            "gate_proj", "up_proj", "down_proj",   # LLM MLP
            "fc1", "fc2",                         # Vision Encoder MLP
            "attn.qkv", "attn.proj"              # Vision Encoder Attention
        ]


    peft_config = LoraConfig(
        r=r_config,
        lora_alpha=r_config * 2,
        lora_dropout=0,
        target_modules=targets,
        bias='none'
    )

    peft_model = get_peft_model(model, peft_config)

    peft_model.print_trainable_parameters()

    # processor.image_processor.size = {"height": 512, "width": 512} # POTENTIALLY PROBLEMATIC
    processor.image_processor.size = {"longest_edge": 512}
    #endregion

    #region Load Data
    if job_type == "lab_lora":    
        train_set = load_dataset("hamzamooraj99/AgriPath-LF16-30k-LAB", split='train')
        val_set = load_dataset("hamzamooraj99/AgriPath-LF16-30k-LAB", split='validation')
    elif job_type == "field_lora":
        train_set = load_dataset("hamzamooraj99/AgriPath-LF16-30k-FIELD", split='train')
        val_set = load_dataset("hamzamooraj99/AgriPath-LF16-30k-FIELD", split='validation')
    else:
        train_set = load_dataset("hamzamooraj99/AgriPath-LF16-30k", split='train')
        val_set = load_dataset("hamzamooraj99/AgriPath-LF16-30k", split='validation')

    smolCollator = SmolCollator(processor)
    #endregion

    #region Fine-tune the Model
    trainer = SFTTrainer(
        model=peft_model,
        tokenizer=processor.tokenizer,
        data_collator=smolCollator,
        train_dataset=train_set,
        eval_dataset=val_set,
        args=SFTConfig(
            # Optimisation & Mixed Precision
            per_device_train_batch_size=8,  #Each GPU processes 2 samples per batch,
            gradient_accumulation_steps=4,  #Gradients are accumulated for 4 steps before updating model
            warmup_steps=100,                #Gradually increases learning rate for first n steps to prevent instability
            num_train_epochs=2,             #Parameter to perform full fine-tune (use max_steps=30 for a quick test)
            learning_rate=learning_rate_config,#wandb.config.learning_rate,
            fp16=not is_bf16_supported(),   #Use float16 if GPU does not support bf16
            bf16=is_bf16_supported(),         #Use bfloat16 if GPU supports it (better stability)
            # Optimiser & Weight Decay
            optim="adamw_8bit",
            weight_decay=weight_decay_config,             #Regularisation to prevent overfitting
            lr_scheduler_type='linear',     #Decay type for learning rate from learning_rate to 0
            seed=3407,
            # Settings for Vision Fine-Tuning
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            # dataset_num_proc=16,             #CPU processes for parallel dataset processing
            # dataloader_num_workers=4,
            # dataloader_persistent_workers=True,
            # max_seq_length=256,
            gradient_checkpointing = False,
            max_grad_norm = 0.3,
            # Validation Settings
            do_eval=True,
            eval_strategy='epoch',
            load_best_model_at_end=False,
            # metric_for_best_model='f1_score',
            # greater_is_better=True,
            per_device_eval_batch_size=8,
            # Logging & Reporting
            report_to='wandb',               #Integration with Weights & Biases ('none' disables, 'wandb' enables)
            run_name=run_name,
            logging_steps=10,
            # Save Settings
            save_strategy='no',
        )
    )
    #endregion

    #region Memory Stats
    #Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    trainer.can_return_loss = True
    trainer_stats = trainer.train()

    #Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")
    #endregion

    run.finish()
    #endregion

if __name__ == '__main__':
    main()