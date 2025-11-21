import os
import torch
import open_clip
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from .utils import ClipContrastiveLoss
from ..utils import OpenClipASCIIDataset as ASCIIDataset, openclip_collate_fn as collate_fn

BATCH_SIZE = 32
LR = 1e-5
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "ViT-B-16"
PRETRAINED = "laion2b_s34b_b88k"
OUTPUT_DIR = "openclip_ascii_finetuned"

CSV_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "data", "dataset", "dataset.csv"
)

model, _, preprocess = open_clip.create_model_and_transforms(
    MODEL_NAME, pretrained=PRETRAINED, device=DEVICE
)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

for param in model.parameters():
    param.requires_grad = True

full_dataset = ASCIIDataset(CSV_PATH)

train_idxs, test_idxs = train_test_split(
    range(len(full_dataset)),
    test_size=0.2,
    stratify=full_dataset.data["caption"],
    random_state=42
)

train_dataset = Subset(full_dataset, train_idxs)
test_dataset = Subset(full_dataset, test_idxs)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    num_workers=4,
    shuffle=True
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    num_workers=4,
    shuffle=False
)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = ClipContrastiveLoss()

def train_epoch(epoch):
    model.train()
    total_loss = 0.0

    for batch in train_dataloader:
        images = torch.stack([preprocess(img) for img in batch["images"]]).to(DEVICE)
        captions = tokenizer(batch["captions"]).to(DEVICE)

        image_features = model.encode_image(images)
        text_features = model.encode_text(captions)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        loss = loss_fn(image_features, text_features)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    return total_loss / len(train_dataloader)

@torch.no_grad()
def evaluate():
    model.eval()
    total_loss = 0.0

    for batch in test_dataloader:
        images = torch.stack([preprocess(img) for img in batch["images"]]).to(DEVICE)
        captions = tokenizer(batch["captions"]).to(DEVICE)

        image_features = model.encode_image(images)
        text_features = model.encode_text(captions)

        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        loss = loss_fn(image_features, text_features)

        total_loss += loss.item()

    return total_loss / len(test_dataloader)

if __name__ == "__main__":
    print("Starting training...")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(epoch)
        val_loss = evaluate()

        print(f"Epoch {epoch}/{EPOCHS} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}")

    print("Training complete. Saving model...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "openclip_ascii.pth"))
    print(f"Model saved to {OUTPUT_DIR}/openclip_ascii.pth")