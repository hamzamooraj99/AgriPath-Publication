# Import Libraries
from unsloth import FastVisionModel
import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import DataLoader
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import pandas as pd
import re as regex
import os
import wandb.errors
import yaml, argparse
import wandb
import string
import random

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

#region CONFIG SETUP
# Argument Parser
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to YAML Config')
args = parser.parse_args()

# Load YAML Config
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)
try:
    artifact_path = config['artifact_path']
except KeyError:
    model_path = config['model_path']
model_name = config['model_name']
run_name = config['run_name']
trc = config['trc']
job_type = config['job_type']
try:
    zs_type = config['zs_type']
except KeyError:
    zs_type = None
try:
    proj_name = config['proj_name']
except KeyError:
    proj_name = "AgriPath-VLM-Sweep-Evals"
#endregion

#region W&B SETUP
# Weights & Biases
# os.environ["WANDB_API_KEY"]=""
wandb.login(key=os.getenv("WANDB_API_KEY"))
run = wandb.init(
    project=proj_name,
    name=run_name,
    config=config,
    job_type=job_type
)
#endregion

#region CROP DISEASE LISTS
crop_diseases = [
    'apple_black_rot', 'apple_cedar_apple_rust', 'apple_fels', 'apple_healthy', 'apple_powdery_mildew', 'apple_rust', 'apple_scab',
    'bell pepper_bacterial_spot', 'bell pepper_healthy', 'bell pepper_leaf_spot',
    'blueberry_healthy',
    'cherry_powdery_mildew', 'cherry_healthy',
    'corn_common_rust', 'corn_gray_leaf_spot', 'corn_leaf_blight', 'corn_healthy', 'corn_nlb', 'corn_phaeosphaeria_leaf_spot',
    'grape_black_measles', 'grape_black_rot', 'grape_healthy', 'grape_leaf_blight',
    'olive_bird_eye_fungus', 'olive_healthy', 'olive_rust_mite',
    'orange_huanglongbing',
    'peach_bacterial_spot', 'peach_healthy',
    'potato_late_blight', 'potato_healthy', 'potato_early_blight',
    'raspberry_healthy',
    'rice_bacterial_leaf_blight', 'rice_bacterial_leaf_streak', 'rice_bacterial_panicle_blight', 'rice_brown_spot', 'rice_dead_heart', 'rice_downy_mildew', 
    'rice_healthy', 'rice_hispa', 'rice_leaf_blast', 'rice_leaf_scald', 'rice_nbls', 'rice_neck_blast', 'rice_tungro',
    'soybean_healthy', 
    'squash_powdery_mildew',
    'strawberry_angular_leaf_spot', 'strawberry_blossom_blight', 'strawberry_gray_mold', 'strawberry_healthy', 'strawberry_leaf_scorch', 'strawberry_leaf_spot', 'strawberry_powdery_mildew', 
    'tomato_bacterial_spot', 'tomato_late_blight', 'tomato_healthy', 'tomato_early_blight', 'tomato_leaf_mold', 'tomato_leaf_spot', 'tomato_mosaic_virus', 'tomato_spider_mites', 
    'tomato_target_spot', 'tomato_yellow_leaf', 
]

# def generate_mcq_keys(count):
#     """Generates a list of MCQ keys (A, B, ..., Z, AA, AB, ...)."""
#     alphabet = string.ascii_uppercase
#     keys = []
#     for i in range(count):
#         if i < 26:
#             keys.append(alphabet[i])
#         else:
#             # For keys beyond Z, start with AA, AB, etc.
#             first_letter = alphabet[(i // 26) - 1]
#             second_letter = alphabet[i % 26]
#             keys.append(f"{first_letter}{second_letter}")
#     return keys

#mcq_keys = generate_mcq_keys(len(crop_diseases))
mcq_keys = ['A', 'B', 'C', 'D']
#endregion

#region DATA COLLATOR
class VisionDataCollator:
    def __init__(self, processor, zs_type=None):
        self.processor = processor
        self.instruction = "You are an expert plant pathologist. Identify the crop and the disease (if any) present in the image provided."
        self.zs_pure_instruction = (
            "You are an expert plant pathologist. Identify the crop and the disease present in the image provided. "
            "Respond in the following format:\n"
            "Class: [crop]\n"
            "Disease: [disease]\n"
        )
        self.zs_context_instruction = f"You are an expert plant pathologist. You have a list of crop-disease pairs and need to identify the crop-disease present in the image provided, by selecting a crop-disease pair from the list. Here is the list:\n{crop_diseases}\nRespond with only the selected crop-disease pair from the list and nothing else."
        self.zs_type = zs_type
    
    def __call__(self, batch):
        images = [sample['image'] for sample in batch]
        labels = [sample['crop_disease_label'] for sample in batch]
        messages = []
        mcq_dicts = []

        for i in range(len(batch)):
            if(self.zs_type == "pure"): 
                instruct = self.zs_pure_instruction
            elif(self.zs_type == "context"): 
                instruct = self.zs_context_instruction
            elif(self.zs_type == "mcq"): 
                true_label = labels[i]
                distractors = [label for label in crop_diseases if label != true_label]
                chosen_distractors = random.sample(distractors, 3)
                options = chosen_distractors + [true_label]
                random.shuffle(options)
                mcq_options_dict = {
                    key: disease 
                    for key, disease in zip(mcq_keys, options)
                }
                mcq_dicts.append(mcq_options_dict)
                zs_mcq_instruction = f"You are an expert plant pathologist. The image shows a plant with a disease.\nWhich of the following is the correct diagnosis?\n{mcq_options_dict}\nRespond with only the letter corresponding to the correct option."
                instruct = zs_mcq_instruction
            else:
                instruct = self.zs_pure_instruction

            messages.append([
                {"role": "system",
                    "content": [
                        {"type": "text", "text": instruct}
                        ]
                },
                {"role": "user",
                "content": [
                        {"type": "text", "text": "Identify the crop and disease in the image."},
                        {"type": "image"}
                    ]
                }
            ])

        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(
            images, input_text,
            add_special_tokens=False,
            truncation=True,
            padding=True,
            return_tensors="pt"
        )

        if 'attention_mask' in inputs:
            inputs['attention_mask'] = inputs['attention_mask'].to(torch.bfloat16)

        return {'inputs': inputs, 'label': labels, 'mcq': mcq_dicts}
#endregion

#region REGEX EXTRACTION
def output_extraction(output_batch):
    patterns = [
        regex.compile(r"(?:Class|Answer|Crop):\s*(\w+(?: \w+)*)\s*[\r\n]+Disease:\s*(\w+(?:_\w+)*)", flags=regex.IGNORECASE),
        regex.compile(r"Answer:\s*[\r\n]+(\w+(?: \w+)*)\s*[\r\n]+(\w+(?:_\w+)*)", flags=regex.IGNORECASE),
        regex.compile(r"Disease:\s*(\w+(?:_\w+)*)\s*[\r\n]+(?:Crop|Class|Answer):\s*(\w+(?: \w+)*)", flags=regex.IGNORECASE)
    ]

    def kv_fallback(output):
        crop = None
        disease = None
        lines = output.splitlines()
        for line in lines:
            if "crop" in line.lower():
                crop_match = regex.search(r"crop\W*[:=]?\W*['\"]?([a-zA-Z]+(?: [a-zA-Z]+)*)['\"]?", line, flags=regex.IGNORECASE)
                if crop_match:
                    crop = crop_match.group(1)
            if "disease" in line.lower():
                disease_match = regex.search(r"disease\W*[:=]?\W*['\"]?([a-zA-Z]+(?:_[a-zA-Z]+)*)['\"]?", line, flags=regex.IGNORECASE)
                if disease_match:
                    disease = disease_match.group(1)
        if crop and disease:
            return f"{crop.lower()}_{disease}"
        return "false_parse"

    def extract(output):
        for pattern in patterns:
            match = pattern.search(output)
            if match:
                group1 = match.group(1)
                group2 = match.group(2)
                if '_' in group1:
                    return f"{group2.lower()}_{group1}", False
                else:
                    return f"{group1.lower()}_{group2}", False
        fallback_res = kv_fallback(output)
        if fallback_res is not None:
            return fallback_res, True
        return "false_parse", False
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(extract, output_batch))
    
    preds, fall_back_flags = zip(*results)
    return preds, fall_back_flags

def output_extraction_zs_pure(output_batch):
    pattern = regex.compile(
        r"^\s*(?:Class|Crop):\s*\*?([\w\s-]+?)\*?\s*[\r\n]+"
        r"Disease:\s*\*?(.*?)(?:\s*\(.*\))?\*?\s*$",
        flags=regex.IGNORECASE | regex.DOTALL
    )
    def extract(output):
        match = pattern.search(output)
        if match:
            crop = match.group(1).strip()
            disease = match.group(2).strip()

            if crop in disease:
                disease = disease.replace(f"{crop} ", '')

            formatted_crop = crop.lower().replace(' ', '_')
            formatted_disease = disease.lower().replace(' ', '_')

            return f"{formatted_crop}_{formatted_disease}", False
        
        return "false_parse", False
    
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(extract, output_batch))
    
    preds, fallback_flags = zip(*results)
    return preds, fallback_flags

#endregion

#region EVALUATION
def eval(data_loader, model, processor, label_idx, eval_batch, zs_type=None):
    #region ====Eval Setup
    # Overall Metrics (Macro Avg)
    bal_acc = Accuracy(task='multiclass', num_classes=66, average='macro').to(DEVICE)
    pr = Precision(task='multiclass', num_classes=66, average='macro').to(DEVICE)
    re = Recall(task='multiclass', num_classes=66, average='macro').to(DEVICE)
    f1 = F1Score(task='multiclass', num_classes=66, average='macro').to(DEVICE)
    cm = ConfusionMatrix(task='multiclass', num_classes=66).to(DEVICE)

    # Per-Class Metrics
    f1_pClass = F1Score(task='multiclass', num_classes=66, average='none').to(DEVICE)
    pr_pClass = Precision(task='multiclass', num_classes=66, average='none').to(DEVICE)
    re_pClass = Recall(task='multiclass', num_classes=66, average='none').to(DEVICE)
    #endregion

    #region ====Eval Loop
    # NEW: Setup for qualitative sampling
    sample_size = 100
    parse_samples = []
    parse_count = 0
    
    fallback_parses = 0
    false_parse_count = 0
    failed_raw_outputs = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating {eval_batch}", unit="batch"):

            all_preds = []
            all_trues = []
            
            if isinstance(batch, tuple):
                batch = {'inputs': batch[0], 'labels': batch[1]}

            inputs = batch['inputs'].to(DEVICE)
            labels = batch['label']

            gen_ids = model.generate(**inputs, max_new_tokens=25, use_cache=True)
            gen_ids = gen_ids[:, inputs.input_ids.shape[1]:]
            gen_texts = processor.batch_decode(gen_ids, skip_special_tokens=True)

            if(zs_type == "pure"):
                predicted_labels, fb_flags = output_extraction_zs_pure(gen_texts)
            elif(zs_type == "context"):
                predicted_labels = gen_texts
            elif(zs_type == "mcq"):
                mcq_dicts = batch['mcq']
                predicted_labels = gen_texts
            else:
                predicted_labels, fb_flags = output_extraction(gen_texts)

            # DEBUGGING BLOCK ==============================================
            # if idx == 0: # Only print for the first batch
            #     print("\n--- DEBUGGING FIRST BATCH ---")
            #     for i in range(len(gen_texts)):
            #         true_label_str = labels[i]
            #         pred_label_str = predicted_labels[i]
                    
            #         print(f"\nSample {i}:")
            #         print(f"  RAW MODEL OUTPUT: {repr(gen_texts[i])}") # 1. Check model output
            #         print(f"  EXTRACTED PRED: '{pred_label_str}'")     # 2. Check regex result
                    
            #         is_in_dict = pred_label_str in label_idx if pred_label_str else False
            #         print(f"  EXTRACTED PRED IN label_idx: {is_in_dict}") # 3. Check if it's a valid key
            #         print(f"  TRUE LABEL: '{true_label_str}'")
            #     print("\n--- END DEBUGGING ---")
            # =============================================================

            for idx, pred_label in enumerate(predicted_labels):
                true_label = labels[idx]

                if zs_type == "context":
                    new_label = "false_parse"
                    for label in crop_diseases:
                        if label in pred_label:
                            new_label = label
                            break

                elif zs_type == "mcq":
                    mcq_dict = mcq_dicts[idx]
                    clean_key = pred_label.strip().upper()
                    new_label = mcq_dict.get(clean_key, "false_parse")
                
                else:
                    new_label = pred_label

                if new_label and new_label in label_idx:
                    try:
                        if fb_flags[idx]:
                            fallback_parses+=1
                    except NameError:
                        pass
                    all_preds.append(label_idx[new_label])
                    all_trues.append(label_idx[true_label])

                    if new_label == "false_parse":
                        false_parse_count += 1
                        failed_raw_outputs.append(repr(gen_texts[idx]))
                
                final_parsed_str = repr(gen_texts[idx])
                parse_tuple = (label_idx[true_label], final_parsed_str)
                if parse_count < sample_size:
                    parse_samples.append(parse_tuple)
                    parse_count += 1

            y_true = torch.tensor(all_trues, device=DEVICE)
            y_pred = torch.tensor(all_preds, device=DEVICE)

            if y_pred.size() == y_true.size() and y_pred.size(0) > 0 and y_true.size(0) > 0:
                bal_acc.update(y_pred, y_true)
                pr.update(y_pred, y_true)
                re.update(y_pred, y_true)
                f1.update(y_pred, y_true)
                cm.update(y_pred, y_true)
                f1_pClass.update(y_pred, y_true)
                pr_pClass.update(y_pred, y_true)
                re_pClass.update(y_pred, y_true)
            
            # if idx == 0:
            #     break
    
    total_samples = len(data_loader.dataset)
    balanced_accuracy = bal_acc.compute().cpu()
    precision = pr.compute().cpu()
    recall = re.compute().cpu()
    f1_score = f1.compute().cpu()
    parse_success_rate = (total_samples - false_parse_count)/total_samples
    conf_mat = cm.compute().cpu()
    per_class_f1_scores = f1_pClass.compute().cpu()
    per_class_pr_scores = pr_pClass.compute().cpu()
    per_class_re_scores = re_pClass.compute().cpu()

    return({
        # 'total_samples': total_samples, 
        'balanced_accuracy': balanced_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'false_parse_count': false_parse_count,                                        
        'parse_success_rate': parse_success_rate,
        'fallback_parse_count': fallback_parses,
        'conf_mat': conf_mat,
        'per_class_f1_scores': per_class_f1_scores,
        'per_class_pr_scores': per_class_pr_scores,
        'per_class_re_scores': per_class_re_scores,
        'parse_sample_list': parse_samples,
        'failed_raw_outputs': failed_raw_outputs,
    })
    #endregion 
#endregion 


#region MAIN
def main():
    #region ====Dataset Prep
    # Load Dataset
    test_set = load_dataset("hamzamooraj99/AgriPath-LF16-30k", split='test').shuffle(seed=42)

    # Separate dataset via source
    field_set = test_set.filter(lambda sample: sample['source']=='field', num_proc=8).shuffle(seed=42)
    lab_set = test_set.filter(lambda sample: sample['source']=='lab', num_proc=8).shuffle(seed=42)

    # Label Mappings
    label_idx = {sample['crop_disease_label']: sample['numeric_label'] for sample in test_set}
    label_idx["false_parse"] = 65
    idx_label = {v: k for k, v in label_idx.items()}
    #endregion

    #region ====Model Prep
    try:
        artifact = run.use_artifact(artifact_path, type='model')
        artifact_dir = artifact.download()
        model_source = artifact_dir
    except Exception:  # Or a specific error
        model_source = model_path
    

    model, processor = FastVisionModel.from_pretrained(
        model_name=model_source,
        load_in_4bit=False,
        load_in_8bit=True
    )

    if(model_name == "AgriPath-Qwen2.5-VL-3B" or model_name == "AgriPath-Qwen2.5-VL-7B"):
        processor.image_processor.do_resize=True
        processor.image_processor.max_pixels=512*512
        processor.image_processor.min_pixels=224*224
    

    FastVisionModel.for_inference(model)
    #endregion

    #region ====Data Loader
    def create_dataloader(dataset, zs_type, batch_size=8):
        collator = VisionDataCollator(processor, zs_type)
        return DataLoader(dataset, batch_size=batch_size, collate_fn=collator, num_workers=4, persistent_workers=True)
    
    main_loader = create_dataloader(test_set, zs_type)
    lab_loader = create_dataloader(lab_set, zs_type)
    field_loader = create_dataloader(field_set, zs_type)
    #endregion

    #region =====Run Eval
    main_metrics = eval(main_loader, model, processor, label_idx, 'main', zs_type)
    lab_metrics = eval(lab_loader, model, processor, label_idx, 'lab', zs_type)
    field_metrics = eval(field_loader, model, processor, label_idx, 'field', zs_type)
    #endregion

    #region ====Log Metrics
    def plot_conf_matrix(conf_mat, eval_batch):
        conf_mat = conf_mat.cpu().numpy()
        fig, ax = plt.subplots(figsize=(14, 14)) # Make figure larger
        ax.matshow(conf_mat, cmap=plt.cm.Blues, alpha=0.3)

        # Only add text if matrix is small (e.g., <= 20 classes)
        if conf_mat.shape[0] <= 20: 
            for i in range(conf_mat.shape[0]):
                for j in range(conf_mat.shape[1]):
                    ax.text(x=j, y=i, s=conf_mat[i, j], va='center', ha='center')
        
        # Add all 66 class labels to axes
        class_labels = [idx_label[i] for i in range(66)]
        ax.set_xticks(range(66))
        ax.set_yticks(range(66))
        ax.set_xticklabels(class_labels, rotation=90, fontsize=6) # Smaller font
        ax.set_yticklabels(class_labels, fontsize=6) # Smaller font

        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f"{run_name}_{eval_batch}")
        plt.tight_layout() # Adjust layout to prevent label cutoff
        return fig
    
    # Log Overall Metrics
    summary_data = [
        ["Main", main_metrics['balanced_accuracy'], main_metrics['precision'], main_metrics['recall'], main_metrics['f1_score'], main_metrics['parse_success_rate']],
        ["Lab", lab_metrics['balanced_accuracy'], lab_metrics['precision'], lab_metrics['recall'], lab_metrics['f1_score'], lab_metrics['parse_success_rate']],
        ["Field", field_metrics['balanced_accuracy'], field_metrics['precision'], field_metrics['recall'], field_metrics['f1_score'], field_metrics['parse_success_rate']]
    ]
    summary_columns = ["Source", "Balanced Accuracy", "Precision", "Recall", "F1 Score", "Parse Success Rate"]
    summary_table = wandb.Table(data=summary_data, columns=summary_columns)

    wandb.log({
        "overall_metrics/Balanced Accuracy": wandb.plot.bar(summary_table, "Source", "Balanced Accuracy"),
        "overall_metrics/Precision": wandb.plot.bar(summary_table, "Source", "Precision"),
        "overall_metrics/Recall": wandb.plot.bar(summary_table, "Source", "Recall"),
        "overall_metrics/F1 Score": wandb.plot.bar(summary_table, "Source", "F1 Score"),
        "overall_metrics/Parse Success Rate": wandb.plot.bar(summary_table, "Source", "Parse Success Rate"),
    })

    # Log Parse Metrics Table Artifacts
    # artifact_name = "parse_metrics_table"
    # try:
    #     artifact = run.use_artifact(f'{artifact_name}: latest')
    #     parse_metrics_table = artifact.get("parse_metrics")
    # except wandb.errors.CommError:
    #     columns = [
    #         "Model",
    #         "False Parse Count (Main)", "Fallback Parse Count (Main)",
    #         "False Parse Count (Lab)","Fallback Parse Count (Lab)",
    #         "False Parse Count (Field)", "Fallback Parse Count (Field)"
    #     ]
    #     parse_metrics_table = wandb.Table(columns=columns)
    
    # parse_metrics_table.add_data(
    #     run_name,
    #     main_metrics['false_parse_count'], main_metrics['fallback_parse_count'],
    #     lab_metrics['false_parse_count'], lab_metrics['fallback_parse_count'],
    #     field_metrics['false_parse_count'], field_metrics['fallback_parse_count'],
    # )

    # new_artifact = wandb.Artifact(artifact_name, type='evaluation_results')
    # new_artifact.add(parse_metrics_table, "parse_metrics")
    # run.log_artifact(new_artifact)
    
    # Log Confusion Matrix
    main_fig = plot_conf_matrix(main_metrics['conf_mat'], 'main')
    lab_fig = plot_conf_matrix(lab_metrics['conf_mat'], 'lab')
    field_fig = plot_conf_matrix(field_metrics['conf_mat'], 'field')
    wandb.log({
        f"confusion_matrix/{run_name}/main": wandb.Image(main_fig),
        f"confusion_matrix/{run_name}/field": wandb.Image(field_fig),
        f"confusion_matrix/{run_name}/lab": wandb.Image(lab_fig),
    })
    plt.close(main_fig); plt.close(lab_fig); plt.close(field_fig)

    # Log Per-Class Metrics
    class_names = [idx_label[i] for i in range(66)]

    combined_per_class_data = {
        "Class Name": class_names,
        "F1 (Main)": main_metrics["per_class_f1_scores"].numpy(),
        "F1 (Field)": field_metrics["per_class_f1_scores"].numpy(),
        "F1 (Lab)": lab_metrics["per_class_f1_scores"].numpy(),
        "Precision (Main)": main_metrics["per_class_pr_scores"].numpy(),
        "Precision (Field)": field_metrics["per_class_pr_scores"].numpy(),
        "Precision (Lab)": lab_metrics["per_class_pr_scores"].numpy(),
        "Recall (Main)": main_metrics["per_class_re_scores"].numpy(),
        "Recall (Field)": field_metrics["per_class_re_scores"].numpy(),
        "Recall (Lab)": lab_metrics["per_class_re_scores"].numpy(),
    }
    df = pd.DataFrame(combined_per_class_data)
    per_class_table = wandb.Table(dataframe=df)
    wandb.log({
        f"per_class_metrics/{run_name}": per_class_table,
    })

    # Log Failed Raw Outputs
    main_fail_table = wandb.Table(columns=["Failed Raw Output"])
    for output in main_metrics['failed_raw_outputs']: main_fail_table.add_data(output)
    wandb.log({
        "false_parses/main_failures": main_fail_table
    })

    # Log Parse Comparisons
    main_parse_comp = wandb.Table(columns=["True Label", "Parsed Output"])
    for true, output in main_metrics['parse_sample_list']: main_parse_comp.add_data(true, output)
    wandb.log({
        "parse_comparison": main_parse_comp
    })


    wandb.finish()
    #endregion
#endregion

if __name__ == '__main__':
    main()