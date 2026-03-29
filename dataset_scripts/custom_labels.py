'''
# custom_labels.py
## Author: @hamzamooraj99 (Hamza Hassan Mooraj)
Description: This file contains a script to include custom labels to the dataset
'''

from datasets import load_dataset, DatasetDict, Dataset

hf_repo = "hamzamooraj99/AgriPath-LF16-30k"

print("Loading dataset...")
dataset = load_dataset(hf_repo)

def custom_label(split_set: Dataset, split: str):
    print(f"Creating custom labels for {split} set")

    crop_disease_label = [(f"{sample['crop'].lower()}_{sample['disease']}") for sample in split_set]
    unique_labels = sorted(set(crop_disease_label))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    numeric_label = [label_map[label] for label in crop_disease_label]

    split_set = split_set.add_column(name='crop_disease_label', column=crop_disease_label)
    split_set = split_set.add_column(name='numeric_label', column=numeric_label)
    print(f"Created custom labels for {split} set\n")
    return split_set

print("Processing dataset...")
processed_dataset = DatasetDict({
    "train": custom_label(dataset['train'], 'train'),
    "test": custom_label(dataset['test'], 'test'),
    "validation": custom_label(dataset['validation'], 'val')
})

train_count = len(processed_dataset['train'])
test_count = len(processed_dataset['test'])
val_count = len(processed_dataset['validation'])

print(f"Train has {train_count} samples")
print(f"Test has {test_count} samples")
print(f"Val has {val_count} samples")

processed_dataset.push_to_hub("hamzamooraj99/AgriPath-CNN")