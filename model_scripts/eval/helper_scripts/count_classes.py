from datasets import load_dataset
from collections import Counter

def count_all_classes(dataset_name="hamzamooraj99/AgriPath-LF16-30K"):
    """
    Counts how many samples belong to each class across *all* splits.
    Assumes each sample has 'numeric_label' and 'crop_disease_label'.
    """
    print(f"📦 Loading dataset: {dataset_name} ...")
    dataset = load_dataset(dataset_name)

    total_counts = Counter()
    label_name_map = {}

    # Go through every split and count
    for split_name, split_data in dataset.items():
        print(f"🔹 Processing split: {split_name}")
        for sample in split_data:
            label_idx = sample["numeric_label"]
            label_name = sample["crop_disease_label"]
            total_counts[label_idx] += 1
            label_name_map[label_idx] = label_name

    # Print the combined totals
    print("\n📊 === CLASS COUNTS (All Splits Combined) ===")
    print(f"{'Label Index':<12} {'Label Name':<40} {'Count':>10}")
    print("-" * 65)
    for idx, count in sorted(total_counts.items()):
        name = label_name_map[idx]
        print(f"{idx:<12} {name:<40} {count:>10}")
    print("-" * 65)
    print(f"Grand Total Samples: {sum(total_counts.values())}\n")

    return total_counts


if __name__ == "__main__":
    count_all_classes()
