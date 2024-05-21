from pathlib import Path
from .. import LanguageModel
from ..engine import VllmEngine

class Llama3(LanguageModel):
    def __init__(self, models_folder:Path, name:str='Meta-Llama-3-8B-Instruct',
                 use_system_prompt:bool=True, chat_template:str=None, 
                 device:str='cuda', engineType=VllmEngine):
        super().__init__(models_folder=models_folder, name=name, use_system_prompt=use_system_prompt, 
                         chat_template=chat_template, device=device, engineType=engineType)

class Llama3_70b(LanguageModel):
    def __init__(self, models_folder:Path, name:str='Meta-Llama-3-70B-Instruct',
                 use_system_prompt:bool=True, chat_template:str=None, 
                 device:str='cuda', engineType=VllmEngine):
        super().__init__(models_folder=models_folder, name=name, use_system_prompt=use_system_prompt, 
                         chat_template=chat_template, device=device, engineType=engineType)
        # TODO does not run as the model needs to be spli over more than one A100
