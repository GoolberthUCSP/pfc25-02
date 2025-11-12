import open_clip
import torch

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16', pretrained='laion2b_s34b_b88k')
tokenizer = open_clip.get_tokenizer('ViT-B-16')
