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
    def query(self, input_string, verbose=False):
        """
        Abstract method for querying the model and getting a response.
        """
        pass

    @abstractmethod
    def get_answer(self, question, chunks, verbose=False):
        """
        Abstract method to get an answer given a question and some chunks passed for context.
        """
        pass

    def is_input_too_long(self, input_string):
        """Returns True if a prompt is too long for the model"""
        return self.token_counter(input_string) >= self.context_size

from .chatgpt import GPT35