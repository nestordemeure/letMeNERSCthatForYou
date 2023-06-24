import re
from abc import ABC, abstractmethod
from typing import List, Dict
from ...database.document_loader import Chunk

def keep_references_only(input_str):
    """keep only lines starting with a *, - or number followed by spaces then a url starting in https or enclosed in <>"""
    pattern = re.compile(r'^(?:\*|-|\d+)\s+(?:https:.*|<https:.*>)$', re.MULTILINE)
    matches = pattern.findall(input_str)
    return '\n'.join(matches)

class LanguageModel(ABC):
    def __init__(self, model_name:str, context_size:int):
        self.model_name = model_name
        self.context_size = context_size

    @abstractmethod
    def token_counter(self, input_string:str) -> int:
        """
        Abstract method for counting tokens in an input string.
        """
        pass

    @abstractmethod
    def query(self, input_string:str, verbose=False) -> str:
        """
        Abstract method for querying the model and getting a response.
        """
        pass

    @abstractmethod
    def get_answer(self, question:str, chunks:List[Chunk], verbose=False) -> str:
        """
        Abstract method to get an answer given a question and some chunks passed for context.
        """
        pass

    @abstractmethod
    def extract_question(self, messages:List[Dict], verbose=False) -> str:
        """
        Abstract method to extract the latest question given a list of messages.
        Message are expectted to be dictionnaries with a 'role' ('user' or 'assistant') and 'content' field.
        the question returned will be a string.
        """
        pass

from .chatgpt import GPT35
from .vicuna import Vicuna
