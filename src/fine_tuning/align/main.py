import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from .utils import AlignLoss
from ..utils import ASCIIDataset, collate_fn
from .trainer import AlignTrainer, training_args
from transformers import AlignModel, AlignProcessor

model = AlignModel.from_pretrained("kakaobrain/align-base", device_map="auto")
processor = AlignProcessor.from_pretrained("kakaobrain/align-base")

for param in model.parameters():
    param.requires_grad = True

# === Paths ===
OUTPUT_DIR = "align_ascii_finetuned"
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "dataset", "dataset.csv")

# === Dataset ===
full_dataset = ASCIIDataset(csv_file=CSV_PATH, processor=processor)
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
loss_fn = AlignLoss()

# === Trainer ===
trainer = AlignTrainer(
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