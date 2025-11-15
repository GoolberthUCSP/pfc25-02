import open_clip
import torch
import numpy as np
from utils import evaluate_model
from transformers.image_utils import load_image

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k', device='cuda')
tokenizer = open_clip.get_tokenizer('ViT-B-16')

# Zero-shot classification function
def classify_image(image_path: str, candidate_labels: list) -> dict:
    image = load_image(image_path)
    image = preprocess(image).unsqueeze(0).to('cuda')
    text = tokenizer(candidate_labels).to('cuda')

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)

    image_features /= image_features.norm(dim=1, keepdim=True)
    text_features /= text_features.norm(dim=1, keepdim=True)

    logits_per_image = image_features @ text_features.T * model.logit_scale.exp()
    probs = logits_per_image.softmax(dim=1).detach().cpu().numpy()[0]

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
    print("Model: OpenCLIP ViT-B/16 (laion2b_s34b_b88k)")
    print(f"Final accuracy: {accuracy * 100:.2f}%")
    print(f"Mean confidence: {confidence * 100:.2f}%")
    print(f"Mean entropy: {entropy * 100:.2f}%")

# Model: OpenCLIP ViT-B/16 (laion2b_s34b_b88k)
# Final accuracy: 49.08%
# Mean confidence: 45.58%
# Mean entropy: 52.57%          