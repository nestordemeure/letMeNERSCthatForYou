import re
from abc import ABC, abstractmethod
from typing import List, Dict
from pathlib import Path
from ...database.document_loader import Chunk

def keep_references_only(input_str):
    # extract all urls
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F])|#)+')
    urls = re.findall(url_pattern, input_str)
    # remove trailing `>` (from <url> types of pattern)
    urls = [url[:-1] if url.endswith('>') else url for url in urls]
    # remove duplicates while preserving order
    urls = list(dict.fromkeys(urls))
    # convert list into a bullet list of URLs
    bullet_list = "\n".join(f"* {url}" for url in urls)
    return bullet_list

class LanguageModel(ABC):
    def __init__(self, models_folder:Path, model_name:str, context_size:int):
        self.models_folder = models_folder
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
