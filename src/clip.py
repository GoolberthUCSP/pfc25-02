import torch
from transformers import AutoProcessor, CLIPModel
from transformers.image_utils import load_image

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")