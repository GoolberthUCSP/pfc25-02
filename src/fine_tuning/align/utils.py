import torch
from torch import nn

class AlignLoss(nn.Module):
    def __init__(self, temperature=0.01):  # Temperature used in ALIGN paper
        super().__init__()
        self.temperature = temperature
        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, outputs):
        # Extract embeddings
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        # L2 normalize
        image_embeds = image_embeds / (image_embeds.norm(dim=-1, keepdim=True) + 1e-6)
        text_embeds  = text_embeds  / (text_embeds.norm(dim=-1, keepdim=True) + 1e-6)

        # Compute logits
        logits = (image_embeds @ text_embeds.t()) / self.temperature

        # Labels are 0..N-1
        batch_size = logits.size(0)
        targets = torch.arange(batch_size, device=logits.device)

        # Symmetric loss (image->text, text->image)
        loss_i = self.loss_fct(logits, targets)
        loss_t = self.loss_fct(logits.t(), targets)

        return (loss_i + loss_t) / 2