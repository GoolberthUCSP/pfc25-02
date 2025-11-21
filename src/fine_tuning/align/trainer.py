from transformers import Trainer, TrainingArguments
import torch

training_args = TrainingArguments(
    output_dir="align_finetuned",
    per_device_train_batch_size=32,   
    per_device_eval_batch_size=32,
    # gradient_accumulation_steps=2,
    num_train_epochs=20,              
    learning_rate=5e-5,               
    eval_strategy="epoch",
    save_strategy="no",
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=50,
    remove_unused_columns=False,
    max_grad_norm=1.0,
    fp16=True,
    report_to="none",
    lr_scheduler_type="cosine",
    warmup_ratio=0.05
)


class AlignTrainer(Trainer):
    def __init__(self, loss_fn, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        outputs = model(**inputs)

        loss = self.loss_fn(outputs)

        return (loss, outputs) if return_outputs else loss