import os
from src.globals import DATASET_CSV_PATH, TMP_PATH
from transformers import AutoProcessor, CLIPModel
from torch.utils.data import Subset
from src.fine_tuning.clip.utils import ClipContrastiveLoss
from src.fine_tuning.utils import ASCIIDataset, collate_fn, get_stratified_indexes
from src.fine_tuning.clip.trainer import CLIPTrainer, training_args

# === Carga modelo y procesador ===
model_dir = "openai/clip-vit-base-patch16"
model = CLIPModel.from_pretrained(model_dir, device_map="auto")
processor = AutoProcessor.from_pretrained(model_dir)

for param in model.parameters():
    param.requires_grad = True

# === Paths ===
OUTPUT_DIR = os.path.join(TMP_PATH, "clip_16_ascii_finetuned")

# === Dataset ===
full_dataset = ASCIIDataset(csv_file=DATASET_CSV_PATH, processor=processor)

train_idxs, test_idxs = get_stratified_indexes()

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
trainer.save_state()

# === Evaluación ===
eval_results = trainer.evaluate()
print(f"Evaluation results: {eval_results}")

# === Guardado final ===
os.makedirs(OUTPUT_DIR, exist_ok=True)
trainer.save_model(output_dir=OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)