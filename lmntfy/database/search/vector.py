from pathlib import Path
from typing import List
from ..chunk import Chunk
from . import SearchEngine

class VectorSearch(SearchEngine):
    """
    Sentence-embedding based vector search.
    Based on [faiss](https://faiss.ai/).
    """
    def __init__(self):
        None

    def _add_chunk(self, chunk_id:int, chunk: Chunk):
        """
        Adds a chunk with the given id.
        """
        pass

    def add_several_chunks(self, chunks: dict[int,Chunk]):
        """
        Adds several chunks with the given indices.
        """
        for (chunk_id, chunk) in chunks:
            self._add_chunk(chunk_id, chunk)

    def _remove_chunk(self, chunk_id:int):
        """
        Remove the chunk wih he given id.
        """
        pass

    def remove_several_chunks(self, chunk_indices: List[int]):
        """
        Removes several chunks from the search engine.
        """
        for chunk_id in chunk_indices:
            self._remove_chunk(chunk_id)
    
    def get_closest_chunks(self, input_text: str, k: int) -> List[(float,int)]:
        """
        Returns the (score,chunk_id) of the closest chunks, in order of decreasing scores.
        """
        pass

    def save(self, database_folder:Path):
        """
        Save the search engine on file.
        """
        pass

    def load(self, database_folder:Path):
        """
        Loads the search engine from file.
        """
        pass