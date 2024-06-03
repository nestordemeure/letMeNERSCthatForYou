from pathlib import Path
from .. import LanguageModel
from ..engine import TransformerEngine, VllmEngine

class Llama3(LanguageModel):
    def __init__(self, models_folder:Path, name:str='Meta-Llama-3-8B-Instruct',
                 use_system_prompt:bool=True, chat_template:str=None, 
                 device:str='cuda', engineType=TransformerEngine):
        super().__init__(models_folder=models_folder, name=name, use_system_prompt=use_system_prompt, 
                         chat_template=chat_template, device=device, engineType=engineType)

class Llama3_70b(LanguageModel):
    def __init__(self, models_folder:Path, name:str='Meta-Llama-3-70B-Instruct',
                 use_system_prompt:bool=True, chat_template:str=None, 
                 device:str='cuda', engineType=VllmEngine):
        # NOTE: enforce the use of VllmEngine
        if engineType != VllmEngine:
            print(f"WARNING: Llama3-70b requires the use of the vLLM engine and the posisbility to spread the weights over several GPUs.")
        # NOTE: keeping memory use low (limiting batch size to 16 but, in practice, expect no more than 2 for realistic interactions)
        engine_kwargs = {'nb_gpus':4, 'enforce_eager':True, 'gpu_memory_utilization':0.95, 'max_model_len':1024*6, 'max_num_seqs':16}
        # creates the model
        super().__init__(models_folder=models_folder, name=name, use_system_prompt=use_system_prompt, 
                         chat_template=chat_template, device=device, engineType=VllmEngine, **engine_kwargs)

class Llama3_70b_awq4bits(LanguageModel):
    """
    4bits AWQ quantization to reduce memory use at the price of some inteligence
    https://huggingface.co/casperhansen/llama-3-70b-instruct-awq
    """
    def __init__(self, models_folder:Path, name:str='llama-3-70b-instruct-awq',
                 use_system_prompt:bool=True, chat_template:str=None, 
                 device:str='cuda', engineType=VllmEngine):
        # NOTE: enforce the use of VllmEngine
        if engineType != VllmEngine:
            print(f"WARNING: Llama3-70b requires the use of the vLLM engine and the posisbility to spread the weights over several GPUs.")
        # NOTE: keeping memory use low (limiting batch size to 16 but, in practice, expect no more than 2 for realistic interactions)
        engine_kwargs = {'nb_gpus':4, 'enforce_eager':True, 'gpu_memory_utilization':0.95, 'max_model_len':1024*6, 'max_num_seqs':16}
        # creates the model
        super().__init__(models_folder=models_folder, name=name, use_system_prompt=use_system_prompt, 
                         chat_template=chat_template, device=device, engineType=VllmEngine, **engine_kwargs)
