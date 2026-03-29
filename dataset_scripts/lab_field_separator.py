from datasets import load_dataset, DatasetDict

def create_lab(full_set):
    print("Filtering lab samples from full dataset...")
    lab_set = full_set.filter(lambda sample: sample['source']=='lab', num_proc=8)
    print("\nCOMPLETED FILTER\nUploading...")
    lab_set.push_to_hub("hamzamooraj99/AgriPath-LF16-30k-LAB")

def create_field(full_set):
    print("Filtering field samples from full dataset...")
    field_set = full_set.filter(lambda sample: sample['source']=='field', num_proc=8)
    print("\nCOMPLETED FILTER\nUploading...")
    field_set.push_to_hub("hamzamooraj99/AgriPath-LF16-30k-FIELD")

if __name__=="__main__":
    print("Loading full dataset...")
    full_set = load_dataset("hamzamooraj99/AgriPath-LF16-30k")
    print("\nCOMPLETED LOADING\n")

    # create_lab(full_set)

    create_field(full_set)

