from pathlib import Path
from typing import List, Dict, Tuple
from abc import ABC, abstractmethod
from ..chunk import Chunk
from ...models import embedding, reranker

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
    def get_closest_chunks(self, input_text: str, chunks:Dict[int,Chunk], k: int) -> List[Tuple[float,int]]:
        """
        Returns the closest chunks to the input text based on similarity scores.

        Parameters:
        input_text (str): The input text to compare against the chunks.
        chunks (Dict[int, Chunk]): A dictionary of chunks where keys are chunk IDs and values are Chunk objects.
        k (int): A lower bound on the number of chunks to return based on their similarity scores.

        Returns:
        List[Tuple[float, int]]: A list of tuples, each containing a similarity score and a chunk ID, ordered from best to worst.
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
from .hybrid import HybridSearch, reciprocal_rank_scores, relative_scores, distribution_based_scores
from .rerank import RerankSearch

# a full fledged default
def Default(models_folder:Path, device='cuda'):
    """
    Classic Hybrid search:
        (Vector,Keyword) search combined with reciprocal_rank_scores
        with a reranker on top
    """
    vector_search = VectorSearch(embedding.Default(models_folder, device=device))
    hybrid_search = HybridSearch(vector_search, KeywordSearch)
    return RerankSearch(reranker.Default(models_folder, device=device), hybrid_search, reciprocal_rank_scores)
