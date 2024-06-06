from pathlib import Path
from typing import List, Tuple, Dict
from ..chunk import Chunk
from . import SearchEngine
from .hybrid import merge_and_sort_scores
from ...models.reranker import Reranker

class RerankSearch(SearchEngine):
    """
    Reranker search augmentation.
    This reorder search results according to a given reranker.
    Useful when not all search result will fit the model's context window.
    You can set k*=2 when using a good reranker.
    See: https://www.pinecone.io/learn/series/rag/rerankers/
    """
    def __init__(self, reranker: Reranker, search_engine: SearchEngine):
        # reranker
        self.reranker: Reranker = reranker
        # search engine we are augmenting
        self.search_engine: SearchEngine = search_engine
        # init parent
        # NOTE: we do not include the reranker in the name as it does not impact the database files
        super().__init__(name=search_engine.name)

    def add_several_chunks(self, chunks: dict[int,Chunk]):
        """
        Adds several chunks with the given indices.
        NOTE: the reranker will break chunks down internally when computing similarities
        """
        self.search_engine.add_several_chunks(chunks)

    def remove_several_chunks(self, chunk_indices: List[int]):
        """
        Removes several chunks from the search engine.
        """
        self.search_engine.remove_several_chunks(chunk_indices)
    
    def get_closest_chunks(self, input_text: str, chunks:Dict[int,Chunk], k: int) -> List[Tuple[float,int]]:
        """
        Returns the (score,chunk_id) of the closest chunks, from best to worst
        """
        # gets the original results
        scored_chunk_ids = self.search_engine.get_closest_chunks(input_text, chunks, k)
        # rerank them
        chunks = [chunks[chunk_id] for (score,chunk_id) in scored_chunk_ids]
        new_scores = self.reranker.similarities(input_text, chunks)
        reranked_chunks = [ (new_score,chunk_id) for (new_score, (score,chunk_id)) in zip(new_scores, scored_chunk_ids)]
        # sort the chunks according to the new score
        reranked_chunks = merge_and_sort_scores(reranked_chunks, merging_strategy=max)
        return reranked_chunks

    def initialize(self, database_folder:Path):
        """
        Initialize the search engine if needed.
        """
        self.search_engine.initialize(database_folder)

    def exists(self, database_folder:Path) -> bool:
        """
        Returns True if an instance of the search engine is saved in the given folder.
        """
        return self.search_engine.exists(database_folder)

    def save(self, database_folder:Path):
        """
        Save the search engine on file.
        """
        self.search_engine.save(database_folder)

    def load(self, database_folder:Path):
        """
        Loads the search engine from file.
        """
        self.search_engine.load(database_folder)
