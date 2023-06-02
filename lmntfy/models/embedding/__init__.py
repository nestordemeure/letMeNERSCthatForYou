from abc import ABC, abstractmethod

class Embedding(ABC):
    def __init__(self, name, embedding_length, tokenizer, max_input_tokens, normalized):
        self.name = name
        self.embedding_length = embedding_length
        self.tokenizer = tokenizer
        self.max_input_tokens = max_input_tokens
        self.normalized = normalized

    @abstractmethod
    def embed(self, text):
        """
        Abstract method for converting text into an embedding.
        """
        pass

from . import openai_embedding