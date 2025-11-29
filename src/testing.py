from transformers import AutoModel, AutoProcessor
from src.globals import DATASET_CSV_PATH, TMP_PATH
from src.fine_tuning.utils import get_stratified_indexes
import torch
import os
from src.utils import evaluate_model
import numpy as np
from transformers.image_utils import load_image


model_name = "siglip"
OUTPUT_DIR = os.path.join(TMP_PATH, f"{model_name}_ascii_finetuned")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cargar modelo y processor
model = AutoModel.from_pretrained(OUTPUT_DIR).to(device)
processor = AutoProcessor.from_pretrained(OUTPUT_DIR)

train_idxs, test_idxs = get_stratified_indexes()

model.eval()  # modo evaluación

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
    accuracy, confidence_gap, entropy = evaluate_model(classify_image, split="subset", indexes=test_idxs)
    print(f"Model               : Pretrained {model_name} fine-tuned")
    print(f"Final accuracy      : {accuracy * 100:.2f}%")
    print(f"Mean confidence gap : {confidence_gap * 100:.2f}%")
    print(f"Mean entropy        : {entropy:.4f}")