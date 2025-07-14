import json

# List of dataset names
data_names = ["food101", "food172", "food2k", "fru92", "veg200", "foodx251"]  # Add more dataset names as needed

# File path templates
base_path = "/map-vepfs/dehua/code/visual-memory"
groundtruth_path_template = f"{base_path}/answers/groundtruth/{{}}_train_groundtruth.jsonl"
attribute_path_template = f"{base_path}/questions/{{}}/train_attribute.jsonl"

# # Description template
# template = """
# I will provide you with an image of a food item belonging to the category: {category}.  
# Your task is to describe the food based solely on the following attributes **without mentioning the category name or type** in any part of the description.

# \nColor - Describe the primary colors and overall hue distribution of the food (e.g., green vegetables, red meat, yellow grains, etc.).
# \nTexture - Describe the surface characteristics of the food (e.g., porous structure of bread, smooth or rough surface of fruit, fibrous texture of meat).
# \nShape - Describe the geometric characteristics of the food (e.g., round fruits, elongated vegetables, irregularly shaped meat chunks).
# \nSize/Proportion - Describe the relative size and proportion of the food.
# \nSurface Features - Describe glossiness (oily or dry appearance), transparency, and reflectivity.

# Finally, summarize the food's visual characteristics in five sentences within 200 word limit. **Ensure that the food's category name ({category}) is NOT mentioned anywhere in your description.**
# """

# Description template
template = """
You will be given an image of a food item from {category}. Provide a concise description (no more than 100 words) covering its color, texture, shape, size/proportion, and surface features. The description should be clear enough to help identify the categpry of the food image, but do not mention {category} anywhere.
"""

# Process each dataset
for data_name in data_names:
    # Define file paths
    file_name = groundtruth_path_template.format(data_name)
    file_name_attribute = attribute_path_template.format(data_name)

    datas = []
    
    # Read ground truth data
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            data = json.loads(line)
            category = data["groundtruth"]
            data["text"] = template.format(category=category)
            datas.append(data)

    # Write processed data to attribute file
    with open(file_name_attribute, 'w', encoding='utf-8') as file:
        for data in datas:
            json.dump(data, file)
            file.write("\n")

    print(f"Processed and saved attribute data for {data_name}")