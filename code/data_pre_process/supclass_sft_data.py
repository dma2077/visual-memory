import json

veg200_filename = "/llm_reco/dehua/code/visual-memory/questions/veg200/dinov2_large_train_5_softmax.json"
veg200_sft_filename = "/llm_reco/dehua/data/food_finetune_data/veg200_supclass_finetune_raw.json"
super_sub_filename = "/llm_reco/dehua/code/Visual-RFT/food_dataset/veg200.json"
with open(super_sub_filename, 'r', encoding='utf-8') as file:
    super_to_sub = json.load(file)

sub_to_super = {
    sub: sup
    for sup, subs in super_to_sub.items()
    for sub in subs
}
conversations = []
with open(veg200_filename, 'r', encoding='utf-8') as file:
    datas = json.load(file)
    for data in datas:
        line = data
        image = line["images"][0].replace("/map-vepfs/dehua/data/data/vegfru-dataset/", "/llm_reco/dehua/data/food_data/")
        category = line["conversations"][1]["value"]
        sup_category = sub_to_super[category]
        conversation = {
        "messages": [
            {
                "content": "<image>What is the category of the food?",
                "role": "user"
            },
            {
                "content": f"{sup_category} | {category}",
                "role": "assistant"
            }
        ],
        "images": [
            image
        ]
    }
        conversations.append(conversation)

with open(veg200_sft_filename, 'w', encoding='utf-8') as file:
    json.dump(conversations, file, indent=2)

