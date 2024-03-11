import torch
from pathlib import Path
from . import LanguageModel

class Qwen(LanguageModel):
    def __init__(self, 
                 models_folder: Path,
                 model_name: str='Qwen1.5-14B-Chat',
                 use_system_prompt=True,
                 device='cuda'):
        super().__init__(models_folder / model_name, use_system_prompt=use_system_prompt, device=device,
                         model_kwargs={'torch_dtype':torch.bfloat16, 'attn_implementation':'flash_attention_2'})
        # NOTE: tokenizer specific
        self.upper_answer_size = 400
        self.upper_question_size = 180
        # NOTE: needed as the full 32k context overflows the GPU memory
        self.context_size = 8*1024
