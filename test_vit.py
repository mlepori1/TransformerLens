from transformers import AutoImageProcessor, ViTForImageClassification
import torch
from datasets import load_dataset
from transformer_lens.HookedViT import HookedViT

dataset = load_dataset("huggingface/cats-image")
image = dataset["test"]["image"][0]

image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
hf_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224").to("mps")
tl_model = HookedViT.from_pretrained("google/vit-base-patch16-224", dtype=torch.float64).to("mps")

inputs = image_processor(image, return_tensors="pt").to("mps")

with torch.no_grad():
    hf_logits = hf_model(**inputs).logits
    tl_logits = tl_model(**inputs)

#print(hf_logits)
#print(tl_logits)

#print(hf_model)
#print(tl_model)

embed_tl = tl_model.embed(**inputs)
embed_hf = hf_model.vit.embeddings(**inputs)

#print(torch.sum(embed_tl != embed_hf))


resid_tl = tl_model.blocks[0](embed_tl)
resid_hf = hf_model.vit.encoder.layer[0](embed_hf)[0]

torch.set_printoptions(precision=10)

print(resid_tl)
print(resid_hf)
print(resid_tl == resid_hf)
print(resid_tl[0][0][0])
print(resid_hf[0][0][0])

print(hf_model.dtype)
print(tl_model.cfg.dtype)

#print(embed_tl.size())
#print(embed_hf.size())
