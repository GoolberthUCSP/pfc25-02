import torch
import numpy as np
from utils import evaluate_model
from transformers import AutoProcessor, CLIPModel
from transformers.image_utils import load_image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", device_map="auto")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

import os
from PIL import Image
from transformers.image_utils import load_image

# Zero-shot classification function
def classify_image(image_path: str, candidate_labels: list) -> dict:
    image = load_image(image_path)
    inputs = processor(
        text=candidate_labels,
        images=image,
        return_tensors="pt",
        padding=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
    
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1).cpu().numpy()[0]
    
    # Confidence score
    confidence_gap = np.abs(probs[0] - probs[1])
    entropy = -np.sum(probs * np.log(probs + 1e-12))

    return {
        "inference": dict(zip(candidate_labels, probs)),
        "confidence_gap": confidence_gap,
        "entropy": entropy
    }

if __name__ == "__main__":
    accuracy, confidence, entropy = evaluate_model(classify_image)
    print("Model: CLIP-ViT-B/32")
    print(f"Final accuracy: {accuracy * 100:.2f}%")
    print(f"Mean confidence: {confidence * 100:.2f}%")
    print(f"Mean entropy: {entropy * 100:.2f}%")

