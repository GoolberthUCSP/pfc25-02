from torch.nn import CrossEntropyLoss
import torch


def collate_fn(batch):
    batch_dict = {}
    keys = set().union(*(d.keys() for d in batch))

    for k in keys:
        try:
            batch_dict[k] = torch.stack([x[k] for x in batch])
        except Exception as e:
            print(f"[WARN] Skipping key {k}: {e}")
    return batch_dict


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
