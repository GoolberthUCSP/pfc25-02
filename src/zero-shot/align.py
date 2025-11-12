import torch
import numpy as np
from utils import evaluate_model
from transformers import AlignModel, AlignProcessor

model = AlignModel.from_pretrained("kakaobrain/align-base", device_map="auto")
processor = AlignProcessor.from_pretrained("kakaobrain/align-base")

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
    
    image_embeds = outputs.image_embeds
    text_embeds = outputs.text_embeds

    image_embeds = image_embeds / image_embeds.norm(dim=1, keepdim=True)
    text_embeds = text_embeds / text_embeds.norm(dim=1, keepdim=True)

    logits_per_image = 100.0 * image_embeds @ text_embeds.T
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
    print("Model: ALIGN-base")
    print(f"Final accuracy: {accuracy * 100:.2f}%")
    print(f"Mean confidence: {confidence * 100:.2f}%")
    print(f"Mean entropy: {entropy * 100:.2f}%")

# Model: ALIGN-base
# Final accuracy: 48.55%
# Mean confidence: 45.66%
# Mean entropy: 53.10%