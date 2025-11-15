import pandas as pd
import os
from torch.utils.data import Dataset
from PIL import Image

class ClipDataset(Dataset):
    def __init__(self, csv_file, processor):
        self.data = pd.read_csv(csv_file)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row["image_path"]
        caption = row["caption"]

        image = Image.open(image_path)

        inputs = self.processor(
            text=caption,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True
        )
        
        inputs = {k: v.squeeze() for k, v in inputs.items()}
        return inputs
