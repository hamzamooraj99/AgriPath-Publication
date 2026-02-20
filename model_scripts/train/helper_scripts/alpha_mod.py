#region Import Libraries
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTConfig, SFTTrainer
import torch
from datasets import load_dataset
from transformers import BitsAndBytesConfig, Idefics3ForConditionalGeneration, AutoProcessor
from peft import LoraConfig, get_peft_model
import wandb
import os
import yaml, argparse
#endregion

os.environ["TOKENIZERS_PARALLELISM"] = "false"

#region Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to YAML Config')
args, unkown = parser.parse_known_args()

#---Load YAML Config
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

model_name = config['model_name']
run_name = config['run_name']
trc = config['trc']
rank = config['r']
lr = config['learning_rate']
wd = config['weight_decay']
version = config['version']
#endregion

#region Weights & Biases Info
os.environ["WANDB_API_KEY"]="9d53025453578d5552c59a417fd34da242216859"
os.environ["WANDB_LOG_MODEL"] = "end"
os.environ["WANDB_PROJECT"] = "AgriPath-VLM"
wandb.login()
#endregion

#region Data Collator
class SmolCollator:
    def __init__(self, processor):
        self.processor = processor
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")
        ]
    
    def __call__(self, examples):
        texts = []
        images = []
        for example in examples:
            image = example['image']
            question = "Identify the crop and disease in the image."
            answer = f"Class: {example['crop']}\nDisease: {example['disease']}"
            messages = [
                {"role": "system", "content": [{"type": "text", "text": "You are an expert pathologist and need to identify the crop and disease present in an image. If it is a healthy crop, classify it as healthy"}]},
                {"role": "user", "content": [{"type": "text", "text": question}, {"type": "image"}]},
                {"role": "assistant", "content": [{"type": "text", "text": answer}]}
            ]
            
            # [FIX] Use tokenize=False to get a string, not token IDs
            text = self.processor.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, # <-- ADD THIS ARGUMENT
                add_generation_prompt=False
            )
            texts.append(text) # No .strip() needed
            images.append([image])
        
        # Now the processor receives strings as expected
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The rest of your code for creating labels will now work
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == self.image_token_id] = -100
        batch["labels"] = labels

        return batch
#endregion


#region Data Formatting
def convert_to_conversation(sample):
    conversation = [
        {"role": "system",
            "content": [
                {"type": "text", "text": "You are an expert pathologist and need to identify the crop and disease present in an image. If it is a healthy crop, classify it as healthy"}
            ]
        },
        {"role": "user",
        "content": [
                {"type": "text", "text": "Identify the crop and disease in the image."},
                {"type": "image", "image": sample['image']}
            ]
        },
        {"role": "assistant",
        "content": [
                {"type": "text", "text": f"Class: {sample['crop']}\nDisease: {sample['disease']}"}
            ]
        }
    ]
    return({"messages": conversation})
#endregion

#region Model Call and Setup
def main():
    wandb.init()

    if version == 'peft':
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=False,
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf4",
            bnb_8bit_compute_dtype=torch.bfloat16
        )

        model = Idefics3ForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        )
        processor = AutoProcessor.from_pretrained(model_name)

        peft_config = LoraConfig(
            r=rank,
            lora_alpha=rank,
            lora_dropout=0,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj", # LLM Attention
                "gate_proj", "up_proj", "down_proj",   # LLM MLP
                "fc1", "fc2",                         # Vision Encoder MLP
                "attn.qkv", "attn.proj"              # Vision Encoder Attention
            ],
            bias='none'
        )
        peft_model = get_peft_model(model, peft_config)
        peft_model.print_trainable_parameters()
        processor.image_processor.size = {"height": 512, "width": 512}
    else:
        model, processor = FastVisionModel.from_pretrained(
            model_name,
            load_in_4bit= False,
            load_in_8bit= True,
            use_gradient_checkpointing="unsloth",
            trust_remote_code=trc,
        )

        peft_model = FastVisionModel.get_peft_model(
            model,
            finetune_vision_layers=True, 
            finetune_attention_modules=True, 
            finetune_language_layers=True, 
            finetune_mlp_modules=True,

            r=rank,
            lora_alpha=rank,
            lora_dropout=0,
            bias='none',
            random_state=3407,
            use_rslora=False,
            loftq_config=None
        )

        # tokenizer = 
        processor.image_processor.do_resize=True
        processor.image_processor.max_pixels=512*512
        processor.image_processor.min_pixels=224*224
    #endregion

    #region Load Data
    train_set = load_dataset("hamzamooraj99/AgriPath-LF16-30k", split='train')
    val_set = load_dataset("hamzamooraj99/AgriPath-LF16-30k", split='validation')
    #endregion

    #region Define Collator
    if version == 'peft':
        collator = SmolCollator(processor)
    else:
        collator = UnslothVisionDataCollator(
            model=model, 
            processor=processor, 
            formatting_func=convert_to_conversation, 
            train_on_responses_only=True,
            instruction_part="user",
            response_part="assistant",
        )
        FastVisionModel.for_training(peft_model)

    #region Fine-tune the Model
    trainer = SFTTrainer(
        model = peft_model,
        processing_class=processor,
        tokenizer=processor.tokenizer,
        data_collator = collator,
        train_dataset = train_set,
        eval_dataset= val_set,
        args = SFTConfig(
            # Optimisation & Mixed Precision
            per_device_train_batch_size=8,  #Each GPU processes 2 samples per batch,
            gradient_accumulation_steps=2,  #Gradients are accumulated for 4 steps before updating model
            warmup_steps=100,                #Gradually increases learning rate for first n steps to prevent instability
            num_train_epochs=2,             #Parameter to perform full fine-tune (use max_steps=30 for a quick test)
            learning_rate=lr,
            fp16=not is_bf16_supported(),   #Use float16 if GPU does not support bf16
            bf16=is_bf16_supported(),         #Use bfloat16 if GPU supports it (better stability)
            # Optimiser & Weight Decay
            optim="adamw_8bit",
            weight_decay=wd,              #Regularisation to prevent overfitting
            lr_scheduler_type='linear',     #Decay type for learning rate from learning_rate to 0
            seed=3407,
            output_dir='outputs',
            # Settings for Vision Fine-Tuning
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            dataset_num_proc=8,             #CPU processes for parallel dataset processing
            dataloader_num_workers=8,
            dataloader_persistent_workers=True,
            max_seq_length=256,
            gradient_checkpointing = False,
            # Validation Settings
            do_eval=True,
            eval_strategy='epoch',
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            per_device_eval_batch_size=4,
            # Logging & Reporting
            report_to='wandb',               #Integration with Weights & Biases ('none' disables, 'wandb' enables)
            run_name=run_name,
            logging_steps=10,
            # Save Settings
            save_strategy='epoch',
            save_total_limit=3
        )
    )
    #endregion

    # os.environ["UNSLOTH_FULLGRAPH"] = '1'

    #region Memory Stats
    #Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")
    if version=='peft': trainer.can_return_loss = True
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

    #region Save the Model
    # --- COMMENT OUT FOR SWEEP RUNS ---
    # model.push_to_hub(save_repo)
    # tokenizer.push_to_hub(save_repo)
    # ----------------------------------
    #endregion

if __name__ == '__main__':
    main()