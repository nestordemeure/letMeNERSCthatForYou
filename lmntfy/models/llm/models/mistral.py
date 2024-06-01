from pathlib import Path
from .. import LanguageModel
from ..engine import TransformerEngine

class Mistral(LanguageModel):
    def __init__(self, models_folder:Path, name:str='Mistral-7B-Instruct-v0.2',
                 use_system_prompt:bool=False, chat_template:str=None, 
                 device:str='cuda', engineType=TransformerEngine):
        super().__init__(models_folder=models_folder, name=name, use_system_prompt=use_system_prompt, 
                         chat_template=chat_template, device=device, engineType=engineType)

class Zephyr(LanguageModel):
    def __init__(self, models_folder:Path, name:str='zephyr-7b-beta',
                 use_system_prompt:bool=False, chat_template:str=None, 
                 device:str='cuda', engineType=TransformerEngine):
        super().__init__(models_folder=models_folder, name=name, use_system_prompt=use_system_prompt, 
                         chat_template=chat_template, device=device, engineType=engineType)

class OpenChat(LanguageModel):
    def __init__(self, models_folder:Path, name:str='openchat-3.5-0106',
                 use_system_prompt:bool=False, chat_template:str=None, 
                 device:str='cuda', engineType=TransformerEngine):
        super().__init__(models_folder=models_folder, name=name, use_system_prompt=use_system_prompt, 
                         chat_template=chat_template, device=device, engineType=engineType)

class Snorkel(LanguageModel):
    def __init__(self, models_folder:Path, name:str='Snorkel-Mistral-PairRM-DPO',
                 use_system_prompt:bool=False, chat_template:str=None, 
                 device:str='cuda', engineType=TransformerEngine):
        super().__init__(models_folder=models_folder, name=name, use_system_prompt=use_system_prompt, 
                         chat_template=chat_template, device=device, engineType=engineType)
        # NOTE: needed as the model's maximum value is too large for the tokenizer, causing nonsense answers
        self.context_size = 2048

class Starling(LanguageModel):
    def __init__(self, models_folder:Path, name:str='Starling-LM-7B-alpha',
                 use_system_prompt:bool=False, chat_template:str=None, 
                 device:str='cuda', engineType=TransformerEngine):
        super().__init__(models_folder=models_folder, name=name, use_system_prompt=use_system_prompt, 
                         chat_template=chat_template, device=device, engineType=engineType)
        # NOTE: needed as 32k context segfaults
        # see: https://huggingface.co/berkeley-nest/Starling-LM-7B-alpha/discussions/25
        self.context_size = 8*1024

class StarlingCode(Starling):
    """variant of the model finetuned for code generation"""
    def __init__(self, models_folder:Path, name:str='Starling-LM-7B-alpha',
                 use_system_prompt:bool=False, chat_template:str=None, 
                 device:str='cuda', engineType=TransformerEngine):
        super().__init__(models_folder=models_folder, name=name, use_system_prompt=use_system_prompt, 
                         chat_template=chat_template, device=device, engineType=engineType)
        # switch template version
        self.tokenizer.tokenizer.chat_template = self.tokenizer.tokenizer.chat_template.replace('GPT4 Correct', 'Code')

class Mixtral(LanguageModel):
    def __init__(self, models_folder:Path, name:str='Mixtral-8x7B-Instruct-v0.1',
                 use_system_prompt:bool=False, chat_template:str=None, 
                 device:str='cuda', engineType=TransformerEngine):
        super().__init__(models_folder=models_folder, name=name, use_system_prompt=use_system_prompt, 
                         chat_template=chat_template, device=device, engineType=engineType)
