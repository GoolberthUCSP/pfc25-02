from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch

class ClipContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        self.loss_fct = CrossEntropyLoss()

    def forward(self, image_features, text_features):
        logits = image_features @ text_features.t() / self.temperature
        labels = torch.arange(len(logits), device=logits.device)

        loss_i = F.cross_entropy(logits, labels)
        loss_t = F.cross_entropy(logits.t(), labels)

        return (loss_i + loss_t) / 2