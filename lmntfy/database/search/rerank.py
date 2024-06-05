from functools import partial
from pathlib import Path
from typing import List
from ..chunk import Chunk
from . import SearchEngine
from .hybrid import merge_and_sort_scores
from ...models.reranker import Reranker

class RerankSearch_raw(SearchEngine):
    """
    Reranker search augmentation.
    This reorder search results according to a given reranker.
    Useful when not all search result will fit the model's context window.
    You can set k*=2 when using a good reranker.
    See: https://www.pinecone.io/learn/series/rag/rerankers/
    """
    def __init__(self, reranker: Reranker, search_engine: SearchEngine, name:str='rescore'):
        # reranker
        self.reranker: Reranker = reranker
        # search engine we are augmenting
        self.search_engine: SearchEngine = search_engine
        # init parent
        super().__init__(name=search_engine.name + name + reranker.name)

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
    
    def get_closest_chunks(self, input_text: str, k: int) -> List[(float,int)]:
        """
        Returns the (score,chunk_id) of the closest chunks, from best to worst
        """
        # gets the original results
        scored_chunks = self.search_engine.get_closest_chunks(input_text, k)
        # rerank them
        chunks = [chunk for (score,chunk) in scored_chunks]
        new_scores = self.reranker.similarities(input_text, chunks)
        reranked_chunks = list(zip(new_scores, chunks))
        # sort the chunks according to the new score
        reranked_chunks = merge_and_sort_scores(reranked_chunks, merging_strategy=max)
        return reranked_chunks

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

# instance that lets you define the reranker / base search engine
RerankSearch = lambda reranker, search_engine: partial(RerankSearch_raw, reranker=reranker, search_engine=search_engine)