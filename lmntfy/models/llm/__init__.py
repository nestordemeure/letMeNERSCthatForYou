from abc import ABC, abstractmethod

class LanguageModel(ABC):
    def __init__(self, model_name, context_size):
        self.model_name = model_name
        self.context_size = context_size

    @abstractmethod
    def token_counter(self, input_string):
        """
        Abstract method for counting tokens in an input string.
        """
        pass

    @abstractmethod
    def query(self, input_string):
        """
        Abstract method for querying the model and getting a response.
        """
        pass

from .chatgpt import GPT35