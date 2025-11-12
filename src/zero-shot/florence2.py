import torch
from transformers import AutoProcessor, AutoModelForCausalLM 

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-base", torch_dtype=torch.float16, trust_remote_code=True)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)