from pathlib import Path
from typing import List
from abc import ABC, abstractmethod
from ..chunk import Chunk

class SearchEngine(ABC):
    """In charge of the search logic."""
    def __init__(self, name:str):
        self.name = name

    @abstractmethod
    def _add_chunk(self, chunk_id:int, chunk: Chunk):
        """
        Abstract method, adds a chunk with the given id.
        """
        pass

    def add_several_chunks(self, chunks: dict[int,Chunk]):
        """
        Adds several chunks with the given indices.
        """
        for (chunk_id, chunk) in chunks:
            self._add_chunk(chunk_id, chunk)

    @abstractmethod
    def _remove_chunk(self, chunk_id:int):
        """
        Abstract method, remove the chunk wih he given id.
        """
        pass

    def remove_several_chunks(self, chunk_indices: List[int]):
        """
        Removes several chunks from the search engine.
        """
        for chunk_id in chunk_indices:
            self._remove_chunk(chunk_id)
    
    @abstractmethod
    def get_closest_chunks(self, input_text: str, k: int) -> List[(float,int)]:
        """
        Abstract method, returns the (score,chunk_id) of the closest chunks, from best to worst
        """
        pass

    @abstractmethod
    def save(self, database_folder:Path):
        """
        Abstract method, save the search engine on file.
        """
        pass

    @abstractmethod
    def load(self, database_folder:Path):
        """
        Abstract method, loads the search engine from file.
        """
        pass

# instances
from .vector import VectorSearch
from .keywords import KeywordSearch
from .hybrid import HybridSearch
from .rerank import RerankSearch