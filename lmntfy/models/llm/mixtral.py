import torch
from pathlib import Path
from . import LanguageModel
from .mistral import MISTRAL_CHAT_TEMPLATE

class Mixtral(LanguageModel):
    def __init__(self, 
                 models_folder: Path,
                 model_name: str='Mixtral-8x7B-Instruct-v0.1',
                 device='cuda'):
        super().__init__(models_folder / model_name, device=device,
                         model_kwargs={'torch_dtype':torch.float16, 'attn_implementation':'flash_attention_2'})
        self.tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE
        self.upper_answer_size = 450
        self.upper_question_size = 200
