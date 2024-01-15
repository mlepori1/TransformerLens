from transformers import AutoProcessor, CLIPVisionModelWithProjection
import torch
from datasets import load_dataset
from transformer_lens.HookedViT import HookedViT
from PIL import Image
import requests

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

image_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
hf_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
tl_model = HookedViT.from_pretrained("openai/clip-vit-base-patch32", is_clip=True)

inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    hf_out = hf_model(**inputs, output_hidden_states=True)
    hf_logits = hf_out.image_embeds
    tl_logits, cache = tl_model.run_with_cache(**inputs)

for i in range(12):
    hf_resid = hf_out.hidden_states[i]
    tl_resid = cache[f"blocks.{i}.hook_resid_pre"]
    print(i)
    assert torch.all(torch.isclose(hf_resid, tl_resid, atol=1e-4))

assert torch.all(torch.isclose(hf_logits, tl_logits, atol=1e-4))
print("tests passed")
