import torch
from torch import nn

def collate_fn(batch):
    batch_dict = {}
    keys = set().union(*(d.keys() for d in batch))

    for k in keys:
        try:
            batch_dict[k] = torch.stack([x[k] for x in batch])
        except Exception as e:
            print(f"[WARN] Skipping key {k}: {e}")
    return batch_dict


class SigLIPLoss(nn.Module):
    def __init__(self, model, temperature=0.1):
        super().__init__()
        self.model = model
        self.temperature = temperature
        self.loss_fct = nn.BCEWithLogitsLoss()

    def forward(self, outputs):
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds

        image_embeds = image_embeds / (image_embeds.norm(p=2, dim=-1, keepdim=True) + 1e-6)
        text_embeds  = text_embeds  / (text_embeds.norm(p=2, dim=-1, keepdim=True) + 1e-6)

        logits = (image_embeds @ text_embeds.t()) / self.temperature
        logits = torch.clamp(logits, -30, 30)

        target = torch.eye(logits.size(0), device=logits.device)

        loss_i = self.loss_fct(logits, target)
        loss_t = self.loss_fct(logits.t(), target)
        loss = (loss_i + loss_t) / 2
        return loss