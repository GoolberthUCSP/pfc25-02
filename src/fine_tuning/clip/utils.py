from torch.nn import CrossEntropyLoss
import torch

class ClipContrastiveLoss(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fct = CrossEntropyLoss()

    def forward(self, outputs):
        logits_per_image = outputs.logits_per_image
        logits_per_text = outputs.logits_per_text

        targets = torch.arange(logits_per_image.size(0)).to(logits_per_image.device)

        loss = (
            self.loss_fct(logits_per_image, targets)
            + self.loss_fct(logits_per_text, targets)
        ) / 2
        return loss
