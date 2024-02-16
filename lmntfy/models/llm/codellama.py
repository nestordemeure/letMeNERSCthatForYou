import torch
from pathlib import Path
from . import LanguageModel
from .llama2 import LLAMA2_CHAT_TEMPLATE

class CodeLlama(LanguageModel):
    def __init__(self, 
                 models_folder: Path,
                 model_name: str='CodeLlama-13b-Instruct-hf',
                 device='cuda'):
        super().__init__(models_folder / model_name, device=device,
                         model_kwargs={'torch_dtype':torch.bfloat16, 'attn_implementation':'flash_attention_2'})
        self.tokenizer.chat_template = LLAMA2_CHAT_TEMPLATE
        self.upper_answer_size = 450
        self.upper_question_size = 200
