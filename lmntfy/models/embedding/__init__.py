from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

class Embedding(ABC):
    def __init__(self, models_folder:Path, name, embedding_length, max_input_tokens, normalized):
        self.models_folder = models_folder
        self.name = name
        self.embedding_length = embedding_length
        self.max_input_tokens = max_input_tokens
        self.normalized = normalized

    def embed(self, text:str) -> np.ndarray:
        """
        Converts text into an embedding.
        """
        try:
            raw_embedding = self._embed(text)
        except Exception as e:
            print(f"An error occurred while embedding the text '{text}': {str(e)}")
            raise  # rethrow the exception after handling
        
        if self.normalized:
            return raw_embedding
        else:
            # normalize the embedding
            norm = np.linalg.norm(raw_embedding)
            if norm == 0: 
                return raw_embedding
            return raw_embedding / norm

    @abstractmethod
    def _embed(self, text:str) -> np.ndarray:
        """
        Abstract method for converting text into an embedding.
        """
        pass

    @abstractmethod
    def count_tokens(self, text:str) -> int:
        """
        Counts the number of tokens used to represent the given text
        """
        pass

from .sbert_embedding import SBERTEmbedding
