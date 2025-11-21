import pandas as pd
import torch
from torch.utils.data import Dataset
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
    return {
        "images": images,
        "captions": captions
    }