from abc import ABC, abstractmethod
from pathlib import Path

class VectorDatabase(ABC):
    def __init__(self, embedding_length, name=None):
        self.embedding_length = embedding_length
        self.name = name

    @abstractmethod
    def add(self, embedding):
        """
        Abstract method for adding a new vector to the database
        returns its index
        """
        pass

    @abstractmethod
    def remove_several(self, indices):
        """
        Abstract method for removing a vector from the database
        """
        pass

    @abstractmethod
    def get_closest(self, input_embedding, k=3):
        """
        Abstract method, returns the indices of the k closest embeddings in the database
        """
        pass

    @abstractmethod
    def save(self, file_path:Path):
        """
        Abstract method for saving the database to a file.
        """
        pass

    @abstractmethod
    def load(self, file_path:Path):
        """
        Abstract method for loading the database from a file.
        """
        pass

from .faiss import FaissVectorDatabase