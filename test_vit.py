from transformers import AutoImageProcessor, ViTForImageClassification
import torch
from datasets import load_dataset
from transformer_lens.HookedViT import HookedViT

dataset = load_dataset("huggingface/cats-image")
image_processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
hf_model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")
tl_model = HookedViT.from_pretrained("google/vit-base-patch16-224")

for i in range(len(dataset["test"])):
    image = dataset["test"]["image"][i]
    inputs = image_processor(image, return_tensors="pt")

    with torch.no_grad():
        hf_out = hf_model(**inputs, output_hidden_states=True)
        hf_logits = hf_out.logits
        tl_logits, cache = tl_model.run_with_cache(**inputs)

    for i in range(12):
        hf_resid = hf_out.hidden_states[i]
        tl_resid = cache[f"blocks.{i}.hook_resid_pre"]
        assert torch.all(torch.isclose(hf_resid, tl_resid, atol=1e-4))

    assert torch.all(torch.isclose(hf_logits, tl_logits, atol=1e-4))
