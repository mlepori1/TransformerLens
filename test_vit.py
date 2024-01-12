from transformers import AutoImageProcessor, ViTForImageClassification
import torch
from datasets import load_dataset
from transformer_lens.HookedViT import HookedViT

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
hf_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
tl_model = HookedViT.from_pretrained("google/vit-base-patch16-224")

inputs = image_processor(image, return_tensors="pt")

with torch.no_grad():
    hf_logits = hf_model(**inputs).logits
    tl_logits = tl_model(**inputs)

print(hf_logits)
print(tl_logits)
