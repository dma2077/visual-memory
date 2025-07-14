from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch

model = AutoModel.from_pretrained("/map-vepfs/models/google/siglip-so400m-patch14-384")
processor = AutoProcessor.from_pretrained("/map-vepfs/models/google/siglip-so400m-patch14-384")

image_path = "/map-vepfs/dehua/data/data/food-101/images/apple_pie/997950.jpg"
image = Image.open(image_path).convert('RGB')
inputs = processor(images=image, return_tensors="pt")


image_encoder = model.vision_model

pixel_values = inputs['pixel_values']
#outputs = image_encoder(pixel_values=batch_pixel_values)
outputs = image_encoder(pixel_values)
print(outputs.last_hidden_state.shape)
# embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()

# texts = ["a photo of 2 cats", "a photo of 2 dogs"]
# inputs = processor(text=texts, images=image, padding="max_length", return_tensors="pt")

# with torch.no_grad():
#     outputs = model(**inputs)

# logits_per_image = outputs.logits_per_image
# probs = torch.sigmoid(logits_per_image) # these are the probabilities
# print(f"{probs[0][0]:.1%} that image 0 is '{texts[0]}'")
