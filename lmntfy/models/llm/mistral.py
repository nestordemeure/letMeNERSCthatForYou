import torch
from pathlib import Path
from . import LanguageModel

class Mistral(LanguageModel):
    def __init__(self, 
                 models_folder: Path,
                 model_name: str='Mistral-7B-Instruct-v0.2',
                 use_system_prompt=False,
                 device='cuda'):
        super().__init__(models_folder / model_name, use_system_prompt=use_system_prompt, device=device,
                         model_kwargs={'torch_dtype':torch.bfloat16, 'attn_implementation':'flash_attention_2'})
        self.upper_answer_size = 450
        self.upper_question_size = 200