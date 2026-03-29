'''
# downsampler.py
## Author: @hamzamooraj99 (Hamza Hassan Mooraj)
Description: This file contains the logic to downsample the dataset and adjust the ratios of lab and field images included. A modular approach to the problem
## Dataset Structure:
'image': This column contains the image file of the crop disease (dtype: Image)
'crop': This is the crop that the sample is representing. This is one of 16 different crops represented in AgriPath-LF16 (dtype: String)
'disease': The disease present on the crop in the sample. There are a total of 41 unique diseases and 63 crop-disease pairs (dtype: String)
'source': This column describes whether the image was taken in a neutral setting (in a lab) or in an environmentally noisy setting (in the field) (dtype: String)
'split': Describes which split from train, test and validation the sample belongs to (dtype: String)
'''

from datasets import load_dataset, Dataset, concatenate_datasets, DatasetDict
import gc

TARGET_PSPLIT = {
    'train': 370,
    'test': 46,
    'validation': 46
}

TARGET_PSOURCE = {split: TARGET_PSPLIT[split]//2 for split in TARGET_PSPLIT}

CROP_DISEASES = [
    ('Apple','black_rot'), ('Apple','cedar_apple_rust'), ('Apple','fels'), ('Apple','healthy'), ('Apple','powdery_mildew'), ('Apple','rust'), ('Apple','scab'),
    ('Bell Pepper','bacterial_spot'), ('Bell Pepper','healthy'), ('Bell Pepper','leaf_spot'),
    ('Blueberry','healthy'),
    ('Cherry','powdery_mildew'), ('Cherry','healthy'),
    ('Corn','common_rust'), ('Corn','gray_leaf_spot'), ('Corn','leaf_blight'), ('Corn','healthy'), ('Corn','nlb'), ('Corn','phaeosphaeria_leaf_spot'),
    ('Grape','black_measles'), ('Grape','black_rot'), ('Grape','healthy'), ('Grape','leaf_blight'),
    ('Olive','bird_eye_fungus'), ('Olive','healthy'), ('Olive','rust_mite'),
    ('Orange','huanglongbing'),
    ('Peach','bacterial_spot'), ('Peach','healthy'),
    ('Potato','late_blight'), ('Potato','healthy'), ('Potato','early_blight'),
    ('Raspberry','healthy'),
    ('Rice','bacterial_leaf_blight'), ('Rice','bacterial_leaf_streak'), ('Rice','bacterial_panicle_blight'), ('Rice','brown_spot'), ('Rice','dead_heart'), ('Rice','downy_mildew'), 
    ('Rice','healthy'), ('Rice','hispa'), ('Rice','leaf_blast'), ('Rice','leaf_scald'), ('Rice','nbls'), ('Rice','neck_blast'), ('Rice','tungro'),
    ('Soybean','healthy'), 
    ('Squash','powdery_mildew'),
    ('Strawberry','angular_leaf_spot'), ('Strawberry','blossom_blight'), ('Strawberry','gray_mold'), ('Strawberry','healthy'), ('Strawberry','leaf_scorch'), ('Strawberry','leaf_spot'), ('Strawberry','powdery_mildew'), 
    ('Tomato','bacterial_spot'), ('Tomato','late_blight'), ('Tomato','healthy'), ('Tomato','early_blight'), ('Tomato','leaf_mold'), ('Tomato','leaf_spot'), ('Tomato','mosaic_virus'), ('Tomato','spider_mites'), 
    ('Tomato','target_spot'), ('Tomato','yellow_leaf') 
]

INDEX = {
    'test' : {
        'Apple': {'black_rot': (0, 61), 'cedar_apple_rust': (62, 88), 'fels': (89, 406), 'healthy': (407, 1041), 'powdery_mildew': (1042, 1159), 'rust': (1160, 1355), 
                  'scab': (1356, 1909)},
        'Bell Pepper': {'bacterial_spot': (1910, 2008), 'healthy': (2009, 2159), 'leaf_spot': (2160, 2167)},
        'Blueberry': {'healthy': (2168, 2328)},
        'Cherry': {'healthy': (2329, 2418), 'powdery_mildew': (2419, 2523)},
        'Corn': {'common_rust': (2524, 2683), 'gray_leaf_spot': (2684, 2901), 'healthy': (2902, 3045), 'leaf_blight': (3046, 3064), 'nlb': (3065, 3396), 
                 'phaeosphaeria_leaf_spot': (3397, 3445)},
        'Grape': {'black_measles': (3446, 3583), 'black_rot': (3584, 3708), 'healthy': (3709, 3757), 'leaf_blight': (3758, 3864)},
        'Olive': {'bird_eye_fungus': (3865, 4264), 'healthy': (4265, 4534), 'rust_mite': (4535, 4834)},
        'Orange': {'huanglongbing': (4835, 5384)},
        'Peach': {'bacterial_spot': (5385, 5613), 'healthy': (5614, 5660)},
        'Potato': {'early_blight': (5661, 5777), 'healthy': (5778, 5792), 'late_blight': (5793, 5912)},
        'Raspberry': {'healthy': (5913, 5960)},
        'Rice': {'bacterial_leaf_blight': (5961, 6222), 'bacterial_leaf_streak': (6223, 6260), 'bacterial_panicle_blight': (6261, 6293), 'brown_spot': (6294, 6673), 
                 'dead_heart': (6674, 6817), 'downy_mildew': (6818, 6879), 'healthy': (6880, 7274), 'hispa': (7275, 7433), 'leaf_blast': (7434, 7915), 
                 'leaf_scald': (7916, 7966), 'nbls': (7967, 8026), 'neck_blast': (8027, 8126), 'tungro': (8127, 8365)},
        'Soybean': {'healthy': (8366, 8880)},
        'Squash': {'powdery_mildew': (8881, 9076)},
        'Strawberry': {'angular_leaf_spot': (9077, 9123), 'blossom_blight': (9124, 9144), 'gray_mold': (9145, 9189), 'healthy': (9190, 9234), 'leaf_scorch': (9235, 9344), 
                       'leaf_spot': (9345, 9406), 'powdery_mildew': (9407, 9457)},
        'Tomato': {'bacterial_spot': (9458, 9680), 'early_blight': (9681, 9788), 'healthy': (9789, 9952), 'late_blight': (9953, 10153), 'leaf_mold': (10154, 10257), 
                   'leaf_spot': (10258, 10449), 'mosaic_virus': (10450, 10491), 'spider_mites': (10492, 10658), 'target_spot': (10659, 10798), 'yellow_leaf': (10799, 11356)},
    },

    'train' : {
        'Apple': {'black_rot': (0, 496), 'cedar_apple_rust': (497, 717), 'fels': (718, 3262), 'healthy': (3263, 8349), 'powdery_mildew': (8350, 9297), 'rust': (9298, 10871), 
                  'scab': (10872, 15312)},
        'Bell Pepper': {'bacterial_spot': (15313, 16111), 'healthy': (16112, 17329), 'leaf_spot': (17330, 17396)},
        'Blueberry': {'healthy': (17397, 18693)},
        'Cherry': {'healthy': (18694, 19424), 'powdery_mildew': (19425, 20266)},
        'Corn': {'common_rust': (20267, 21555), 'gray_leaf_spot': (21556, 23306), 'healthy': (23307, 24465), 'leaf_blight': (24466, 24621), 'nlb': (24622, 27283), 
                 'phaeosphaeria_leaf_spot': (27284, 27678)},
        'Grape': {'black_measles': (27679, 28785), 'black_rot': (28786, 29794), 'healthy': (29795, 30194), 'leaf_blight': (30195, 31056)},
        'Olive': {'bird_eye_fungus': (31057, 33453), 'healthy': (33454, 34879), 'rust_mite': (34880, 36067)},
        'Orange': {'huanglongbing': (36068, 40474)},
        'Peach': {'bacterial_spot': (40475, 42313), 'healthy': (42314, 42691)},
        'Potato': {'early_blight': (42692, 43628), 'healthy': (43629, 43750), 'late_blight': (43751, 44718)},
        'Raspberry': {'healthy': (44719, 45112)},
        'Rice': {'bacterial_leaf_blight': (45113, 47202), 'bacterial_leaf_streak': (47203, 47506), 'bacterial_panicle_blight': (47507, 47777), 'brown_spot': (47778, 50826), 
                 'dead_heart': (50827, 51980), 'downy_mildew': (51981, 52476), 'healthy': (52477, 55642), 'hispa': (55643, 56918), 'leaf_blast': (56919, 60771), 
                 'leaf_scald': (60772, 61233), 'nbls': (61234, 61731), 'neck_blast': (61732, 62531), 'tungro': (62532, 64449)},
        'Soybean': {'healthy': (64450, 68574)},
        'Squash': {'powdery_mildew': (68575, 70147)},
        'Strawberry': {'angular_leaf_spot': (70148, 70492), 'blossom_blight': (70493, 70650), 'gray_mold': (70651, 71035), 'healthy': (71036, 71497), 'leaf_scorch': (71498, 72386), 
                       'leaf_spot': (72387, 72878), 'powdery_mildew': (72879, 73307)},
        'Tomato': {'bacterial_spot': (73308, 75098), 'early_blight': (75099, 75970), 'healthy': (75971, 77285), 'late_blight': (77286, 78903), 'leaf_mold': (78904, 79738), 
                   'leaf_spot': (79739, 81282), 'mosaic_virus': (81283, 81625), 'spider_mites': (81626, 82969), 'target_spot': (82970, 84093), 'yellow_leaf': (84094, 88572)},
    },

    'validation' : {
        'Apple': {'black_rot': (0, 61), 'cedar_apple_rust': (62, 88), 'fels': (89, 406), 'healthy': (407, 1041), 'powdery_mildew': (1042, 1159), 'rust': (1160, 1355), 
                  'scab': (1356, 1909)},
        'Bell Pepper': {'bacterial_spot': (1910, 2008), 'healthy': (2009, 2159), 'leaf_spot': (2160, 2167)},
        'Blueberry': {'healthy': (2168, 2328)},
        'Cherry': {'healthy': (2329, 2418), 'powdery_mildew': (2419, 2523)},
        'Corn': {'common_rust': (2524, 2683), 'gray_leaf_spot': (2684, 2901), 'healthy': (2902, 3045), 'leaf_blight': (3046, 3064), 'nlb': (3065, 3396), 
                 'phaeosphaeria_leaf_spot': (3397, 3445)},
        'Grape': {'black_measles': (3446, 3583), 'black_rot': (3584, 3708), 'healthy': (3709, 3757), 'leaf_blight': (3758, 3864)},
        'Olive': {'bird_eye_fungus': (3865, 4324), 'healthy': (4325, 4554), 'rust_mite': (4555, 4844)},
        'Orange': {'huanglongbing': (4845, 5394)},
        'Peach': {'bacterial_spot': (5395, 5623), 'healthy': (5624, 5670)},
        'Potato': {'early_blight': (5671, 5787), 'healthy': (5788, 5802), 'late_blight': (5803, 5922)},
        'Raspberry': {'healthy': (5923, 5970)},
        'Rice': {'bacterial_leaf_blight': (5971, 6232), 'bacterial_leaf_streak': (6233, 6270), 'bacterial_panicle_blight': (6271, 6303), 'brown_spot': (6304, 6683),
                  'dead_heart': (6684, 6827), 'downy_mildew': (6828, 6889), 'healthy': (6890, 7285), 'hispa': (7286, 7444), 'leaf_blast': (7445, 7927), 'leaf_scald': (7928, 7978), 
                  'nbls': (7979, 8039), 'neck_blast': (8040, 8139), 'tungro': (8140, 8378)},
        'Soybean': {'healthy': (8379, 8893)},
        'Squash': {'powdery_mildew': (8894, 9089)},
        'Strawberry': {'angular_leaf_spot': (9090, 9132), 'blossom_blight': (9133, 9161), 'gray_mold': (9162, 9208), 'healthy': (9209, 9253), 'leaf_scorch': (9254, 9363), 
                       'leaf_spot': (9364, 9424), 'powdery_mildew': (9425, 9477)},
        'Tomato': {'bacterial_spot': (9478, 9700), 'early_blight': (9701, 9808), 'healthy': (9809, 9972), 'late_blight': (9973, 10173), 'leaf_mold': (10174, 10277), 
                   'leaf_spot': (10278, 10469), 'mosaic_virus': (10470, 10511), 'spider_mites': (10512, 10678), 'target_spot': (10679, 10818), 'yellow_leaf': (10819, 11376)},
    }
}

flagged_pairs = []

def downsample_split(ds_split: str):

    print(f"Processing {ds_split} set...")
    agripath = load_dataset("hamzamooraj99/AgriPath-LF16", split=ds_split)

    crop_disease_set = []

    total_samples = 0

    for cd_pair in CROP_DISEASES:

        clear_vars = ["lab_samples", 'field_samples', 'lab_count', 'field_count']

        crop = cd_pair[0]
        disease = cd_pair[1]

        start_idx = INDEX[ds_split][crop][disease][0]
        end_idx = INDEX[ds_split][crop][disease][1]

        print(f"Filtering for CROP: {crop} | DISEASE: {disease}")
        crop_disease = agripath.select(range(start_idx, end_idx))
        lab_samples = crop_disease.filter(lambda sample: sample['source']=='lab')
        field_samples = crop_disease.filter(lambda sample: sample['source']=='field')

        lab_count = len(lab_samples)
        field_count = len(field_samples)

        print(f"Starting downsample... LAB: {lab_count} | FIELD: {field_count}")

        # CASE 1: Samples only exist for one source -> Downsample existing source (Results in no source split)
        if(lab_count == 0 or field_count == 0):
            if(len(crop_disease) <= TARGET_PSPLIT[ds_split] - 10):
                source = "lab" if field_count == 0 else "field"
                flagged_pairs.append((crop, disease, ds_split, source, len(crop_disease), "CASE 1"))
                crop_disease_set.append(crop_disease)
                msg = (f"CASE 1A - {len(crop_disease)}")
                total_samples += len(crop_disease)
            elif(len(crop_disease) < TARGET_PSPLIT[ds_split]+20 and len(crop_disease) > TARGET_PSPLIT[ds_split]-10):
                crop_disease_set.append(crop_disease)
                msg = (f"CASE 1B - {len(crop_disease)}")
                total_samples += len(crop_disease)
            else:
                case1c_set = crop_disease.shuffle(seed=42).select(range(TARGET_PSPLIT[ds_split]))
                crop_disease_set.append(case1c_set)
                msg = (f"CASE 1C - {len(case1c_set)}")
                total_samples += len(case1c_set)
        
        # CASE 2L: Lab sample count is less than target_pSource amount but field sample count is more -> Include entirety of lab samples and downsample field samples to match difference
        elif(lab_count < TARGET_PSOURCE[ds_split]):
            if(field_count + lab_count < TARGET_PSPLIT[ds_split]):
                crop_disease_set.append(crop_disease)
                msg = (f"CASE 2LA - {len(crop_disease)}")
                total_samples += len(crop_disease)
            else:
                field_downsampled = field_samples.shuffle(seed=42).select(range(TARGET_PSPLIT[ds_split] - lab_count))
                crop_disease_set.append(concatenate_datasets([lab_samples, field_downsampled]))
                msg = (f"CASE 2LB - {len(field_downsampled) + lab_count}")
                total_samples += len(field_downsampled) + lab_count
        
        # CASE 2F: Field sample count is less than target_pSource amount but lab sample count is more-> Include entirety of field samples and downsample lab samples to match difference
        elif(field_count < TARGET_PSOURCE[ds_split]):
            if(field_count + lab_count < TARGET_PSPLIT[ds_split]):
                crop_disease_set.append(crop_disease)
                msg = (f"CASE 2FA - {len(crop_disease)}")
                total_samples += len(crop_disease)
            else:
                lab_downsampled = lab_samples.shuffle(seed=42).select(range(TARGET_PSPLIT[ds_split] - field_count))
                crop_disease_set.append(concatenate_datasets([field_samples, lab_downsampled]))
                msg = (f"CASE 2FB - {len(lab_downsampled) + field_count}")
                total_samples += len(lab_downsampled) + field_count
        
        # CASE 3: Both lab and field sample counts are less than target_pSource amont -> Include entirety of both lab and field samples and then flag the group for review
        elif(lab_count < TARGET_PSOURCE[ds_split]-5 and field_count < TARGET_PSOURCE[ds_split]-5):
            flagged_pairs.append((crop, disease, ds_split, len(crop_disease), "CASE 3"))
            crop_disease_set.append(crop_disease)
            msg = (f"CASE 3 - {len(crop_disease)}")
            total_samples += len(crop_disease)
        
        # CASE 4: Both lab and field sample counts are more than or equal to target_pSource amount -> Downsample both lab and field samples (results in 50/50 split between lab and field)
        else:
            lab_downsampled = lab_samples.shuffle(seed=42).select(range(TARGET_PSOURCE[ds_split]))
            field_downsampled = field_samples.shuffle(seed=42).select(range(TARGET_PSOURCE[ds_split]))
            crop_disease_set.append(concatenate_datasets([field_downsampled, lab_downsampled]))
            msg = (f"CASE 4 - {len(field_downsampled) + len(lab_downsampled)}")
            total_samples += len(field_downsampled) + len(lab_downsampled)

        print(f"{msg} for {crop}-{disease} pair... NEXT")
        print("=============================================================================================")
        print("")

    print(f"Pushing {total_samples} samples for {ds_split} to HUB")
    crop_disease_set: Dataset = concatenate_datasets(crop_disease_set)
    crop_disease_set.push_to_hub("hamzamooraj99/AgriPath-LF16-30k", split=ds_split)
    print(f"Pushed to {ds_split} split to Hub")

if __name__ == '__main__':
    downsample_split("train")
    # downsample_split("test")
    # downsample_split("validation")