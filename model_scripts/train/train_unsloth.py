#region Script Setup
#---Import Libraries
from unsloth import FastVisionModel, is_bf16_supported
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTConfig, SFTTrainer
import torch
from datasets import load_dataset
import wandb
import os
import yaml, argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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
r_config = config['r']
try:
    learning_rate_config = config['learning_rate']
    weight_decay_config = config['weight_decay']
except KeyError:
    pass

#---Weights & Biases Info
# os.environ["WANDB_API_KEY"]="[WANDB_API_KEY]"
os.environ["WANDB_LOG_MODEL"] = "end"
os.environ["WANDB_PROJECT"] = "AgriPath-VLM-Sweep"
wandb.login(key=os.getenv("WANDB_API_KEY"))
#endregion


#region Data Formatting
def convert_to_conversation(sample):
    conversation = [
        {"role": "system",
            "content": [
                # This is the line you need to change
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
    run = wandb.init(
        # name=run_name,
        config=config,
        job_type=job_type
    )

    learning_rate_config = wandb.config.learning_rate
    weight_decay_config = wandb.config.weight_decay

    if job_type == "train_frozen_vision":
        ft_vis = False
    else:
        ft_vis = True

    model, tokenizer = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit= False,
        load_in_8bit= True,
        use_gradient_checkpointing="unsloth",
        trust_remote_code=trc,
    )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=ft_vis, # CHANGE FOR FROZEN VISION
        finetune_attention_modules=True, 
        finetune_language_layers=True, 
        finetune_mlp_modules=True,

        r=r_config,
        lora_alpha=r_config * 2,
        lora_dropout=0,
        bias='none',
        random_state=3407,
        use_rslora=False,
        loftq_config=None
    )

    # tokenizer = 
    tokenizer.image_processor.do_resize=True
    tokenizer.image_processor.max_pixels=512*512
    tokenizer.image_processor.min_pixels=224*224
    #endregion

    if job_type == "field_lora":
        dataset_repo = "hamzamooraj99/AgriPath-LF16-30k-FIELD"
    elif job_type == "lab_lora":
        dataset_repo = "hamzamooraj99/AgriPath-LF16-30k-LAB"
    else:
        dataset_repo = "hamzamooraj99/AgriPath-LF16-30k"

    #region Load Data
    train_set = load_dataset(dataset_repo, split='train')
    val_set = load_dataset(dataset_repo, split='validation')
    #endregion

    #region Fine-tune the Model
    FastVisionModel.for_training(model)
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        data_collator = UnslothVisionDataCollator(
            model=model, 
            processor=tokenizer, 
            formatting_func=convert_to_conversation, 
            train_on_responses_only=True,
            instruction_part="user",
            response_part="assistant",
        ),
        train_dataset = train_set,
        eval_dataset= val_set,
        args = SFTConfig(
            # Optimisation & Mixed Precision
            per_device_train_batch_size=8,  #Each GPU processes 2 samples per batch,
            gradient_accumulation_steps=4,  #Gradients are accumulated for 4 steps before updating model
            warmup_steps=100,                #Gradually increases learning rate for first n steps to prevent instability
            num_train_epochs=2,             #Parameter to perform full fine-tune (use max_steps=30 for a quick test)
            learning_rate=learning_rate_config,
            fp16=not is_bf16_supported(),   #Use float16 if GPU does not support bf16
            bf16=is_bf16_supported(),         #Use bfloat16 if GPU supports it (better stability)
            # Optimiser & Weight Decay
            optim="adamw_8bit",
            weight_decay=weight_decay_config,              #Regularisation to prevent overfitting
            lr_scheduler_type='linear',     #Decay type for learning rate from learning_rate to 0
            seed=3407,
            # output_dir='/pv-cache/qwen7_field_lora',
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
            load_best_model_at_end=False,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            per_device_eval_batch_size=8,
            # Logging & Reporting
            report_to='wandb',               #Integration with Weights & Biases ('none' disables, 'wandb' enables)
            run_name=run_name,
            logging_steps=10,
            # Save Settings
            save_strategy='no',
            # save_total_limit=1,
            # save_safetensors=True,
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
    trainer_stats = trainer.train()
    print(">>> trainer.train() returned, entering post-train section", flush=True)


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
    print(">>> calling run.finish()", flush=True)
    #endregion


if __name__ == '__main__':
    main()