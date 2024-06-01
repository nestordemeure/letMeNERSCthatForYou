from pathlib import Path
from .. import LanguageModel
from ..engine import TransformerEngine

class Gemma(LanguageModel):
    def __init__(self, models_folder:Path, name:str='gemma-7b-it',
                 use_system_prompt:bool=False, chat_template:str=None, 
                 device:str='cuda', engineType=TransformerEngine):
        super().__init__(models_folder=models_folder, name=name, use_system_prompt=use_system_prompt, 
                         chat_template=chat_template, device=device, engineType=engineType)
