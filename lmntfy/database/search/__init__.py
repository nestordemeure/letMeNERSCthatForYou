from pathlib import Path
from typing import List, Dict, Tuple
from abc import ABC, abstractmethod
from ..chunk import Chunk
from ...models import embedding, reranker

#----------------------------------------------------------------------------------------
# ABSTRACT CLASS

class SearchEngine(ABC):
    """In charge of the search logic."""
    def __init__(self, name:str):
        self.name = name

    @abstractmethod
    def add_several_chunks(self, chunks: dict[int,Chunk]):
        """
        Adds several chunks with the given indices.
        """
        pass

    @abstractmethod
    def remove_several_chunks(self, chunk_indices: List[int]):
        """
        Removes several chunks from the search engine.
        """
        pass
    
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
    def initialize(self, database_folder:Path):
        """
        Initialize the search engine if needed.
        """
        pass

    @abstractmethod
    def exists(self, database_folder:Path) -> bool:
        """
        Returns True if an instance of the search engine is saved in the given folder.
        """
        pass

    @abstractmethod
    def save(self, database_folder:Path):
        """
        Save the search engine on file.
        """
        pass

    @abstractmethod
    def load(self, database_folder:Path):
        """
        Loads the search engine from file. Does nothing if it does no exist.
        """
        pass

#----------------------------------------------------------------------------------------
# INSTANCES

# building blocks
from .vector import VectorSearch
from .keywords import KeywordSearch
from .hybrid import HybridSearch, reciprocal_rank_scores, relative_scores, distribution_based_scores
from .rerank import RerankSearch

def Just_Keyword(models_folder:Path, device='cuda'):
    return KeywordSearch()

def Just_Vector(models_folder:Path, device='cuda'):
    return VectorSearch(embedding.Default(models_folder, device='cuda'))

def Reranked_Vectors(models_folder:Path, device='cuda'):
    vector_search = VectorSearch(embedding.Default(models_folder, device=device))
    return RerankSearch(reranker.TFIDFReranker(models_folder, device=device), vector_search)

def Just_Hybrid(models_folder:Path, device='cuda'):
    vector_search = VectorSearch(embedding.Default(models_folder, device='cuda'))
    keyword_search = KeywordSearch()
    return HybridSearch(vector_search, keyword_search, distribution_based_scores)

def Reranked_Hybrid(models_folder:Path, device='cuda'):
    vector_search = VectorSearch(embedding.Default(models_folder, device='cuda'))
    keyword_search = KeywordSearch()
    hybrid_search = HybridSearch(vector_search, keyword_search, reciprocal_rank_scores)
    return RerankSearch(reranker.TFIDFReranker(models_folder, device=device), hybrid_search)

# our current default
Default = Reranked_Hybrid