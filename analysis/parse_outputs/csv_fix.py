from pathlib import Path
import pandas as pd
import os

SCRIPT_DIR = Path(__file__).resolve().parent

print("Current working directory:", os.getcwd())
print("Script directory:", SCRIPT_DIR)
print("Files in script directory:", [p.name for p in SCRIPT_DIR.iterdir() if p.is_file()])

label_names = [
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

files = ["qwen3.csv", "qwen7.csv", "smol.csv"]

for file in files:
    path = SCRIPT_DIR / file

    if not path.exists():
        raise FileNotFoundError(f"Could not find {file} at: {path}")
    
    df = pd.read_csv(path)

    df["True Label (Text)"] = df["True Label"].map(lambda x: label_names[int(x)])

    df = df[["True Label", "True Label (Text)", "Parsed Output"]]
    df.to_csv(path, index=False)
