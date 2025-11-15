import os
from transformers import AutoProcessor, AutoModel
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from utils import collate_fn, SigLIPLoss
from dataset import SigLIPDataset
from trainer import SigLIPTrainer, training_args
import torch

# === Carga modelo y procesador ===
model = AutoModel.from_pretrained("google/siglip-base-patch16-224", device_map="auto")
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224") 

for param in model.parameters():
    param.requires_grad = True

# === Paths ===
OUTPUT_DIR = "siglip_ascii_finetuned"
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "dataset", "dataset.csv")

# === Dataset ===
full_dataset = SigLIPDataset(csv_file=CSV_PATH, processor=processor)
data_df = full_dataset.data

train_idxs, test_idxs = train_test_split(
    range(len(full_dataset)),
    test_size=0.2,
    stratify=data_df["caption"],
    random_state=42
)

train_dataset = Subset(full_dataset, train_idxs)
test_dataset = Subset(full_dataset, test_idxs)

# === Loss ===
loss_fn = SigLIPLoss(model=model)

# === Trainer ===
trainer = SigLIPTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=processor,
    loss_fn=loss_fn,
    args=training_args,
    data_collator=collate_fn,
)

# === Entrenamiento ===
trainer.train()

# === Guardado final ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(output_dir=OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)