"""
0            apple_black_rot                                 492
1            apple_cedar_apple_rust                          272
2            apple_fels                                      462
3            apple_healthy                                   462
4            apple_powdery_mildew                            462
5            apple_rust                                      462
6            apple_scab                                      462
7            bell pepper_bacterial_spot                      462
8            bell pepper_healthy                             462
9            bell pepper_leaf_spot                            80
10           blueberry_healthy                               462
11           cherry_healthy                                  462
12           cherry_powdery_mildew                           462
13           corn_common_rust                                462
14           corn_gray_leaf_spot                             462
15           corn_healthy                                    462
16           corn_leaf_blight                                191
17           corn_nlb                                        462
18           corn_phaeosphaeria_leaf_spot                    466
19           grape_black_measles                             462
20           grape_black_rot                                 462
21           grape_healthy                                   462
22           grape_leaf_blight                               462
23           olive_bird_eye_fungus                           462
24           olive_healthy                                   462
25           olive_rust_mite                                 462
26           orange_huanglongbing                            462
27           peach_bacterial_spot                            462
28           peach_healthy                                   462
29           potato_early_blight                             462
30           potato_healthy                                  149
31           potato_late_blight                              462
32           raspberry_healthy                               462
33           rice_bacterial_leaf_blight                      462
34           rice_bacterial_leaf_streak                      377
35           rice_bacterial_panicle_blight                   334
36           rice_brown_spot                                 462
37           rice_dead_heart                                 462
38           rice_downy_mildew                               492
39           rice_healthy                                    462
40           rice_hispa                                      462
41           rice_leaf_blast                                 462
42           rice_leaf_scald                                 470
43           rice_nbls                                       489
44           rice_neck_blast                                 462
45           rice_tungro                                     462
46           soybean_healthy                                 462
47           squash_powdery_mildew                           462
48           strawberry_angular_leaf_spot                    432
49           strawberry_blossom_blight                       205
50           strawberry_gray_mold                            474
51           strawberry_healthy                              458
52           strawberry_leaf_scorch                          462
53           strawberry_leaf_spot                            491
54           strawberry_powdery_mildew                       472
55           tomato_bacterial_spot                           462
56           tomato_early_blight                             462
57           tomato_healthy                                  462
58           tomato_late_blight                              462
59           tomato_leaf_mold                                462
60           tomato_leaf_spot                                462
61           tomato_mosaic_virus                             424
62           tomato_spider_mites                             462
63           tomato_target_spot                              462
64           tomato_yellow_leaf                              462
"""

class_count = {
    "apple_black_rot": 492,
    "apple_cedar_apple_rust": 272,
    "apple_fels": 462,
    "apple_healthy": 462,
    "apple_powdery_mildew": 462,
    "apple_rust": 462,
    "apple_scab": 462,
    "bell_pepper_bacterial_spot": 462,
    "bell_pepper_healthy": 462,
    "bell_pepper_leaf_spot": 80,
    "blueberry_healthy": 462,
    "cherry_healthy": 462,
    "cherry_powdery_mildew": 462,
    "corn_common_rust": 462,
    "corn_gray_leaf_spot": 462,
    "corn_healthy": 462,
    "corn_leaf_blight": 191,
    "corn_nlb": 462,
    "corn_phaeosphaeria_leaf_spot": 466,
    "grape_black_measles": 462,
    "grape_black_rot": 462,
    "grape_healthy": 462,
    "grape_leaf_blight": 462,
    "olive_bird_eye_fungus": 462,
    "olive_healthy": 462,
    "olive_rust_mite": 462,
    "orange_huanglongbing": 462,
    "peach_bacterial_spot": 462,
    "peach_healthy": 462,
    "potato_early_blight": 462,
    "potato_healthy": 149,
    "potato_late_blight": 462,
    "raspberry_healthy": 462,
    "rice_bacterial_leaf_blight": 462,
    "rice_bacterial_leaf_streak": 377,
    "rice_bacterial_panicle_blight": 334,
    "rice_brown_spot": 462,
    "rice_dead_heart": 462,
    "rice_downy_mildew": 492,
    "rice_healthy": 462,
    "rice_hispa": 462,
    "rice_leaf_blast": 462,
    "rice_leaf_scald": 470,
    "rice_nbls": 489,
    "rice_neck_blast": 462,
    "rice_tungro": 462,
    "soybean_healthy": 462,
    "squash_powdery_mildew": 462,
    "strawberry_angular_leaf_spot": 432,
    "strawberry_blossom_blight": 205,
    "strawberry_gray_mold": 474,
    "strawberry_healthy": 458,
    "strawberry_leaf_scorch": 462,
    "strawberry_leaf_spot": 491,
    "strawberry_powdery_mildew": 472,
    "tomato_bacterial_spot": 462,
    "tomato_early_blight": 462,
    "tomato_healthy": 462,
    "tomato_late_blight": 462,
    "tomato_leaf_mold": 462,
    "tomato_leaf_spot": 462,
    "tomato_mosaic_virus": 424,
    "tomato_spider_mites": 462,
    "tomato_target_spot": 462,
    "tomato_yellow_leaf": 462
}

# Find the key with the maximum value
max_key = max(class_count, key=class_count.get)

print(f"The key with the maximum value is: {max_key}")
print(f"The maximum value is: {class_count[max_key]}")