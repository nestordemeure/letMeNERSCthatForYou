from abc import ABC
from pathlib import Path
from typing import List, Dict
from .engine import LLMEngine, TransformerEngine
from ..tokenizer import ChatTokenizer

class LanguageModel(ABC):
    """
    Large language model.
    Combines an engine and the corresponding chat tokenizer.
    """
    def __init__(self, models_folder:Path, name:str, use_system_prompt: bool, 
                 chat_template:str=None, device:str='cuda',
                 engineType=TransformerEngine, **engine_kwargs):
        # names of the things
        self.name = name
        self.pretrained_model_name_or_path = str(models_folder / name)
        # loads the components of the model
        self.engine: LLMEngine = engineType(self.pretrained_model_name_or_path, device=device, **engine_kwargs)
        self.tokenizer = ChatTokenizer(self.pretrained_model_name_or_path, context_size=self.engine.context_size, 
                                       chat_template=chat_template, use_system_prompt=use_system_prompt)
        # parameters of the LLM
        self.context_size = self.engine.context_size
        self.upper_answer_size = self.tokenizer.upper_answer_size
        self.upper_question_size = self.tokenizer.upper_question_size
        self.device = device

    def count_tokens(self, text:str) -> int:
        """
        Counts the number of tokens in a given string.
        """
        return self.tokenizer.count_tokens(text)

    def apply_chat_template(self, messages: List[Dict[str, str]], nb_tokens_max:int=None) -> str:
        """
        Takes a list of messages and applies the model's chat template.

        NOTE:
        - drops optional messages until the result fits in the given size
        - merge all systems messages into the first message
        """
        return self.tokenizer.apply_chat_template(messages, nb_tokens_max)

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
        return await self.engine.generate(prompt, stopwords, strip_stopword, verbose)

from .models import *