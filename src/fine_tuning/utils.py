import os
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
from torch.utils.data import Dataset
from src.globals import DATASET_IDXS_PATH, DATASET_CSV_PATH, TEST_SIZE
from PIL import Image

class ASCIIDataset(Dataset):
    def __init__(self, csv_file, processor):
        self.data = pd.read_csv(csv_file)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row["image_path"]
        caption = row["caption"]

        image = Image.open(image_path).convert("RGB")

        inputs = self.processor(
            text=caption,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )
        
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        return inputs
    

class OpenClipASCIIDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row["image_path"]
        caption = row["caption"]

        image = Image.open(image_path).convert("RGB")

        return {
            "image": image,
            "caption": caption
        }
        

def collate_fn(batch):
    batch_dict = {}
    keys = set().union(*(d.keys() for d in batch))

    for k in keys:
        try:
            batch_dict[k] = torch.stack([x[k] for x in batch])
        except Exception as e:
            print(f"[WARN] Skipping key {k}: {e}")
    return batch_dict

def openclip_collate_fn(batch):
    images = [item["image"] for item in batch]
    captions = [item["caption"] for item in batch]
    return images, captions

def get_stratified_indexes(test_size=TEST_SIZE, random_state=42):
    # 1. Si el archivo existe, intentamos leerlo
    if os.path.exists(DATASET_IDXS_PATH):
        try:
            df = pd.read_csv(DATASET_IDXS_PATH)

            if "index" in df.columns and "split" in df.columns:
                train_idxs = df[df["split"] == "train"]["index"].tolist()
                test_idxs  = df[df["split"] == "test"]["index"].tolist()
                
                print(f"Indexes loaded from: {DATASET_IDXS_PATH}")
                return train_idxs, test_idxs

            else:
                print("[WARN] CSV found, but without valid columns")

        except Exception as e:
            print(f"[WARN] Error reading {DATASET_IDXS_PATH}: {e}. Regenerating...")

    print("Generating stratified indexes...")

    full_dataset = ASCIIDataset(csv_file=DATASET_CSV_PATH, processor=None)
    data_df = full_dataset.data

    train_idxs, test_idxs = train_test_split(
        range(len(full_dataset)),
        test_size=test_size,
        stratify=data_df["caption"],
        random_state=random_state
    )

    df = pd.DataFrame({
        "index": list(train_idxs) + list(test_idxs),
        "split": ["train"] * len(train_idxs) + ["test"] * len(test_idxs)
    })

    df.to_csv(DATASET_IDXS_PATH, index=False)
    print(f"New indexes saved to: {DATASET_IDXS_PATH}")

    return train_idxs, test_idxs