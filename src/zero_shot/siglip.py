import torch
import numpy as np
from src.utils import evaluate_model
from transformers.image_utils import load_image
from transformers import AutoProcessor, AutoModel

model_dir = "google/siglip-base-patch16-224"
model = AutoModel.from_pretrained(model_dir, device_map="auto", attn_implementation="sdpa")
processor = AutoProcessor.from_pretrained(model_dir) 

# Zero-shot classification function
def classify_image(image_path: str, candidate_labels: list) -> dict:
    image = load_image(image_path).convert("RGB")
    inputs = processor(images=image, 
        text=candidate_labels, 
        return_tensors="pt",
        padding=True).to(model.device)

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
    accuracy, confidence_gap, entropy = evaluate_model(classify_image)
    print("Model: SigLIP ViT-B/16")
    print(f"Final accuracy      : {accuracy * 100:.2f}%")
    print(f"Mean confidence gap : {confidence_gap * 100:.2f}%")
    print(f"Mean entropy        : {entropy:.4f}")