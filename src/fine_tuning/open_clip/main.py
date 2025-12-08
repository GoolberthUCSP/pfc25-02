import os
import torch
import open_clip
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from .utils import ClipContrastiveLoss
from ..utils import OpenClipASCIIDataset as ASCIIDataset, openclip_collate_fn as collate_fn

BATCH_SIZE = 64
LR = 1e-5
EPOCHS = 20
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
    shuffle=True
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    collate_fn=collate_fn,
    shuffle=False
)

optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
loss_fn = ClipContrastiveLoss()

def train_epoch(epoch):
    model.train()
    total_loss = 0.0

    for images, captions in train_dataloader:
        images = torch.stack([preprocess(img) for img in images]).to(DEVICE)
        captions = tokenizer(captions).to(DEVICE)

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

    for images, captions in test_dataloader:
        images = torch.stack([preprocess(img) for img in images]).to(DEVICE)
        captions = tokenizer(captions).to(DEVICE)

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
    print(f"Model saved to {OUTPUT_DIR}/openclip_ascii.pth") # Capas de tensión

#     C:\Users\fred\AppData\Roaming\Python\Python312\site-packages\timm\models\layers\__init__.py:48: FutureWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
#   warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", FutureWarning)
# Starting training...
# Epoch 1/20 - Train Loss: 4.0867 - Val Loss: 3.9398
# Epoch 2/20 - Train Loss: 3.8643 - Val Loss: 3.8093
# Epoch 3/20 - Train Loss: 3.7003 - Val Loss: 3.8040
# Epoch 4/20 - Train Loss: 3.6086 - Val Loss: 3.8221
# Epoch 5/20 - Train Loss: 3.5664 - Val Loss: 3.8388
# Epoch 6/20 - Train Loss: 3.5440 - Val Loss: 3.8345
# Epoch 7/20 - Train Loss: 3.5316 - Val Loss: 3.8774
# Epoch 8/20 - Train Loss: 3.5182 - Val Loss: 3.8274
# Epoch 9/20 - Train Loss: 3.5101 - Val Loss: 3.8673
# Epoch 10/20 - Train Loss: 3.5207 - Val Loss: 3.8492
# Epoch 11/20 - Train Loss: 3.4978 - Val Loss: 3.8321
# Epoch 12/20 - Train Loss: 3.5042 - Val Loss: 3.8689
# Epoch 13/20 - Train Loss: 3.5021 - Val Loss: 3.8260
# Epoch 14/20 - Train Loss: 3.4942 - Val Loss: 3.8514
# Epoch 15/20 - Train Loss: 3.5019 - Val Loss: 3.9008
# Epoch 16/20 - Train Loss: 3.4913 - Val Loss: 3.9298
# Epoch 17/20 - Train Loss: 3.4927 - Val Loss: 3.9317
# Epoch 18/20 - Train Loss: 3.4943 - Val Loss: 3.9744
# Epoch 19/20 - Train Loss: 3.4984 - Val Loss: 3.8911
# Epoch 20/20 - Train Loss: 3.4831 - Val Loss: 3.9058
# Training complete. Saving model...
# Model saved to openclip_ascii_finetuned/openclip_ascii.pth