import torch
from pathlib import Path
from . import LanguageModel

class Starling(LanguageModel):
    def __init__(self, 
                 models_folder: Path,
                 model_name: str='Starling-LM-7B-alpha',
                 use_system_prompt=False,
                 device='cuda'):
        super().__init__(models_folder / model_name, use_system_prompt=use_system_prompt, device=device,
                         model_kwargs={'torch_dtype':torch.bfloat16, 'attn_implementation':'flash_attention_2'})
        self.upper_answer_size = 450
        self.upper_question_size = 200
        # NOTE: needed as 32k context segfaults
        # see: https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha/discussions/25
        self.context_size = 8*1024

class StarlingCode(Starling):
    """variant of the model finetuned for code generation"""
    def __init__(self, 
                 models_folder: Path,
                 model_name: str='Starling-LM-7B-alpha',
                 use_system_prompt=False,
                 device='cuda'):
        super().__init__(models_folder, use_system_prompt=use_system_prompt, device=device)
        # switch template version
        self.tokenizer.chat_template = self.tokenizer.chat_template.replace('GPT4 Correct', 'Code')
        