import os
from transformers import AutoProcessor, CLIPModel
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset
from utils import collate_fn, ClipContrastiveLoss
from dataset import ClipDataset
from trainer import CLIPTrainer, training_args

# === Carga modelo y procesador ===
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", device_map="auto")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

for param in model.parameters():
    param.requires_grad = True

# === Paths ===
OUTPUT_DIR = "clip_ascii_finetuned"
CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "dataset", "dataset.csv")

# === Dataset ===
full_dataset = ClipDataset(csv_file=CSV_PATH, processor=processor)
data_df = full_dataset.data

train_idxs, test_idxs = train_test_split(
    range(len(full_dataset)),
    test_size=0.2,
    stratify=data_df["caption"],
    random_state=42
)

train_dataset = Subset(full_dataset, train_idxs)
test_dataset = Subset(full_dataset, test_idxs)

# === Función de pérdida personalizada ===
loss_fn = ClipContrastiveLoss(model=model)

# === Trainer ===
trainer = CLIPTrainer(
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