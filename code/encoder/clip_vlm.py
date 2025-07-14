from PIL import Image
import requests
from transformers import AutoTokenizer
from transformers import CLIPProcessor, CLIPModel

from transformers import CLIPVisionModel
from transformers import CLIPModel
import torch
from typing import Optional

class CustomCLIPModel(CLIPModel):
    def __init__(self, config):
        super().__init__(config)
        # Add any custom initialization here
        # For example, initialize additional layers or parameters
    def get_image_features(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        return_dict: Optional[bool] = None,
    ) -> torch.FloatTensor:
        """
        Returns:
            image_features (`torch.FloatTensor` of shape `(batch_size, output_dim`): The image embeddings obtained by
            applying the projection layer to the pooled output of [`CLIPVisionModel`].
        """
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        pooled_output = vision_outputs[1]  # pooled_output
        image_features = self.visual_projection(pooled_output)

        # You can add custom processing here if needed
        # For example, apply a custom transformation to image_features

        return image_features



image_model = CLIPVisionModel.from_pretrained("/mnt/madehua/model/clip-vit-large-patch14-336")



model = CLIPModel.from_pretrained("/mnt/madehua/model/clip-vit-large-patch14-336")

tokenizer = AutoTokenizer.from_pretrained("/mnt/madehua/model/clip-vit-large-patch14-336")
    
text_inputs = tokenizer(["a photo of a cat", "a photo of a dog"], padding=True, return_tensors="pt")
processor = CLIPProcessor.from_pretrained("/mnt/madehua/model/clip-vit-large-patch14-336")




url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

inputs = processor(images=image, return_tensors="pt")
text=["a photo of a cat", "a photo of a dog"]
image_features = model.get_image_features(**inputs)
text_features = model.get_text_features(**text_inputs)
print(image_features.shape)
print(text_features.shape)

inputs = processor(text=text, images=image, return_tensors="pt", padding=True)

# outputs = model(**inputs)
# logits_per_image = outputs.logits_per_image # this is the image-text similarity score
# probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities
# print(probs)