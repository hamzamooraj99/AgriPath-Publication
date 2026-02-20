import regex
import random

# crop_diseases = [
#     'apple_black_rot', 'apple_cedar_apple_rust', 'apple_fels', 'apple_healthy', 'apple_powdery_mildew', 'apple_rust', 'apple_scab',
#     'bell pepper_bacterial_spot', 'bell pepper_healthy', 'bell pepper_leaf_spot',
#     'blueberry_healthy',
#     'cherry_powdery_mildew', 'cherry_healthy',
#     'corn_common_rust', 'corn_gray_leaf_spot', 'corn_leaf_blight', 'corn_healthy', 'corn_nlb', 'corn_phaeosphaeria_leaf_spot',
#     'grape_black_measles', 'grape_black_rot', 'grape_healthy', 'grape_leaf_blight',
#     'olive_bird_eye_fungus', 'olive_healthy', 'olive_rust_mite',
#     'orange_huanglongbing',
#     'peach_bacterial_spot', 'peach_healthy',
#     'potato_late_blight', 'potato_healthy', 'potato_early_blight',
#     'raspberry_healthy',
#     'rice_bacterial_leaf_blight', 'rice_bacterial_leaf_streak', 'rice_bacterial_panicle_blight', 'rice_brown_spot', 'rice_dead_heart', 'rice_downy_mildew', 
#     'rice_healthy', 'rice_hispa', 'rice_leaf_blast', 'rice_leaf_scald', 'rice_nbls', 'rice_neck_blast', 'rice_tungro',
#     'soybean_healthy', 
#     'squash_powdery_mildew',
#     'strawberry_angular_leaf_spot', 'strawberry_blossom_blight', 'strawberry_gray_mold', 'strawberry_healthy', 'strawberry_leaf_scorch', 'strawberry_leaf_spot', 'strawberry_powdery_mildew', 
#     'tomato_bacterial_spot', 'tomato_late_blight', 'tomato_healthy', 'tomato_early_blight', 'tomato_leaf_mold', 'tomato_leaf_spot', 'tomato_mosaic_virus', 'tomato_spider_mites', 
#     'tomato_target_spot', 'tomato_yellow_leaf' 
# ]


# decoded_labels = ['System: You are an expert pathologist and need to identify the crop and disease present in an image. If it is a healthy crop, classify it as healthy\nUser: Identify the crop and disease in the image.\nAssistant: Class: Apple\nDisease: black_rot\n',
# 'System: You are an expert pathologist and need to identify the crop and disease present in an image. If it is a healthy crop, classify it as healthy\nUser: Identify the crop and disease in the image.\nAssistant: Class: Apple\nDisease: black_rot\n',
# 'System: You are an expert pathologist and need to identify the crop and disease present in an image. If it is a healthy crop, classify it as healthy\nUser: Identify the crop and disease in the image.\nAssistant: Class: Apple\nDisease: black_rot\n']

# decoded_labels = list(map(lambda sample: sample.split("Assistant: ")[-1], decoded_labels))

# for i in range(len(decoded_labels)):
#     print(f"Decoded Label #{i}: {repr(decoded_labels[i])}")

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

print(f"Actual: {crop_diseases[29]}\nPredicted: {crop_diseases[31]}")

print(f"1: {crop_diseases[55]}\n2: {crop_diseases[56]}\n2: {crop_diseases[60]}")

dataset_check = [
    "rice_brown_spot", "rice_healthy", "rice_leaf_blast", "rice_neck_blast", #NBCD

    'apple_scab', 'apple_black_rot', 'apple_cedar_apple_rust', 'apple_healthy', 'blueberry_healthy', 'cherry_powdery_mildew', 'cherry_healthy', 'corn_gray_leaf_spot', 'corn_common_rust', 'corn_nlb', 'corn_healthy', 'grape_black_rot', 'grape_black_measles', 'grape_leaf_blight', 'grape_healthy', 'orange_huanglongbing', 'peach_bacterial_spot', 'peach_healthy', 'bell pepper_bacterial_spot', 'bell pepper_healthy', 'potato_late_blight', 'potato_healthy', 'potato_early_blight', 'raspberry_healthy', 'soybean_healthy', 'squash_powdery_mildew', 'strawberry_healthy', 'strawberry_leaf_scorch', 'tomato_bacterial_spot', 'tomato_early_blight', 'tomato_healthy', 'tomato_late_blight', 'tomato_leaf_mold', 'tomato_leaf_spot', 'tomato_spider_mites', 'tomato_target_spot', 'tomato_mosaic_virus', 'tomato_yellow_leaf', #PlantVillage

    'apple_scab', 'apple_rust', 'apple_healthy', 'bell pepper_healthy', 'bell pepper_leaf_spot', 'blueberry_healthy', 'cherry_healthy', 'corn_gray_leaf_spot', 'corn_leaf_blight', 'corn_common_rust', 'grape_black_rot', 'grape_healthy', 'peach_healthy', 'potato_late_blight', 'potato_early_blight', 'raspberry_healthy', 'soybean_healthy', 'squash_powdery_mildew', 'strawberry_healthy', 'tomato_bacterial_spot', 'tomato_late_blight', 'tomato_healthy', 'tomato_early_blight', 'tomato_leaf_mold', 'tomato_leaf_spot', 'tomato_mosaic_virus', 'tomato_spider_mites', 'tomato_yellow_leaf', #PlantDoc

    'apple_fels', 'apple_healthy', 'apple_powdery_mildew', 'apple_rust', 'apple_scab', #Apple Dataset 2021

    'rice_bacterial_leaf_blight', 'rice_nbls', 'rice_leaf_scald', 'rice_leaf_blast', 'rice_healthy', 'rice_brown_spot', #Roboflow

    'rice_bacterial_leaf_blight', 'rice_bacterial_leaf_streak', 'rice_bacterial_panicle_blight', 'rice_leaf_blast', 'rice_brown_spot', 'rice_dead_heart', 'rice_downy_mildew', 'rice_hispa', 'rice_healthy', 'rice_tungro', #PaddyDoctor

    'rice_bacterial_leaf_blight', 'rice_leaf_blast', 'rice_brown_spot', 'rice_tungro', #Rice Leaf Disease

    'corn_gray_leaf_spot', #CD&S Dataset

    'corn_gray_leaf_spot', 'corn_common_rust', 'corn_healthy', 'corn_nlb', 'corn_phaeosphaeria_leaf_spot', #Diseases of Maize

    'corn_nlb', #CornNLB

    'olive_bird_eye_fungus', 'olive_healthy', 'olive_rust_mite', #Zeytin Olive Leaf Disease

    'strawberry_leaf_spot', 'strawberry_powdery_mildew', 'strawberry_gray_mold', 'strawberry_angular_leaf_spot', 'strawberry_blossom_blight', #Strawberry Disease Detection
]

dataset_check = set(dataset_check)
print(dataset_check)
print("\n")
print(len(dataset_check))