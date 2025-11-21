from transformers import AutoModel, AutoProcessor
import torch
import os


model = "clip"
CURR_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURR_DIR,"..", "..", "..", f"{model}_ascii_finetuned")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo y processor
model = AutoModel.from_pretrained(OUTPUT_DIR).to(device)
processor = AutoProcessor.from_pretrained(OUTPUT_DIR)

model.eval()  # modo evaluación

from zero_shot.utils import evaluate_model
import numpy as np
from transformers.image_utils import load_image
import os

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "dataset", "dataset.csv")

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
    accuracy, confidence, entropy = evaluate_model(classify_image)
    print(f"Model: Pretrained {model} fine-tuned")
    print(f"Final accuracy: {accuracy * 100:.2f}%")
    print(f"Mean confidence: {confidence * 100:.2f}%")
    print(f"Mean entropy: {entropy * 100:.2f}%")