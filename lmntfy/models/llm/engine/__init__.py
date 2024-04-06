from abc import ABC, abstractmethod
from typing import List

class LLMEngine(ABC):
    """
    Encapsulates a LLM engine.
    """
    def __init__(self, pretrained_model_name_or_path:str, context_size:int=None, device='cuda'):
        self.pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        self.context_size = context_size
        self.device = device

    @abstractmethod
    async def generate(self, prompt:str, stopwords:List[str]=[], strip_stopword:bool=True, verbose:bool=False) -> str:
        """
        Query the model and get a response.

        Args:
            prompt (str): the text prompt
            stopwords (List[str]): the words on which to stop the generation, if any
            strip_stopword (bool): should we strip the stopword from our output (default to True)
            verbose (bool): should we print debug information? (defaults to False)

        Returns:
            str: The generated response from the model.
        """
        pass

# imports the various engines
from .transformer_engine import TransformerEngine
from .vllm_engine import VllmEngine
