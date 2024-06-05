"""
TODO:
    * <https://news.ycombinator.com/item?id=40524759>
    * <https://www.assembled.com/blog/better-rag-results-with-reciprocal-rank-fusion-and-hybrid-search>
    * <https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/azure-ai-search-outperforming-vector-search-with-hybrid/ba-p/3929167>
    * <https://docs.llamaindex.ai/en/stable/examples/retrievers/relative_score_dist_fusion/>
"""
from functools import partial
from pathlib import Path
from typing import List, Tuple, Callable
from ..chunk import Chunk
from . import SearchEngine

#----------------------------------------------------------------------------------------
# UTILS

# basic addition function
addition = lambda x, y: x+y

def merge_and_sort_scores(scored_chunks: List[Tuple[float, int]], merging_strategy: Callable[[float, float], float] = max) -> List[Tuple[float, int]]:
    """
    Takes a list of (score, chunk_id) and:
    * merges identical chunks using the given merging strategy (addition, max, etc)
    * sorts them from largest to smallest by score
    """
    # Merge identical chunks using the given merging strategy
    chunk_dict = {}
    for score, chunk in scored_chunks:
        if chunk in chunk_dict:
            chunk_dict[chunk] = merging_strategy(chunk_dict[chunk], score)
        else:
            chunk_dict[chunk] = score

    # Convert the dictionary back to a list of tuples
    merged_list = [(score, chunk) for chunk, score in chunk_dict.items()]

    # Sort the merged list by scores in descending order
    sorted_list = sorted(merged_list, key=lambda x: x[0], reverse=True)
    return sorted_list

#----------------------------------------------------------------------------------------
# SEARCH ENGINE

class HybridSearch_raw(SearchEngine):
    """
    Hybird search.
    """
    def __init__(self, search_engine1: SearchEngine, search_engine2: SearchEngine, name:str='hybrid'):
        # search engines we are augmenting
        self.search_engine1: SearchEngine = search_engine1
        self.search_engine2: SearchEngine = search_engine2
        # init parent
        super().__init__(name=name + search_engine1.name + search_engine2.name)

    def add_several_chunks(self, chunks: dict[int,Chunk]):
        """
        Adds several chunks with the given indices.
        """
        self.search_engine1.add_several_chunks(chunks)
        self.search_engine2.add_several_chunks(chunks)

    def remove_several_chunks(self, chunk_indices: List[int]):
        """
        Removes several chunks from the search engine.
        """
        self.search_engine1.remove_several_chunks(chunk_indices)
        self.search_engine2.remove_several_chunks(chunk_indices)
    
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
        reranked_chunks = merge_and_sort_scores(reranked_chunks)
        return reranked_chunks

    def save(self, database_folder:Path):
        """
        Save the search engine on file.
        """
        self.search_engine1.save(database_folder)
        self.search_engine2.save(database_folder)

    def load(self, database_folder:Path):
        """
        Loads the search engine from file.
        """
        self.search_engine1.load(database_folder)
        self.search_engine2.load(database_folder)

# instance that lets you define the reranker / base search engine
HybridSearch = lambda search_engine1, search_engine12: partial(HybridSearch_raw, search_engine1=search_engine1, search_engine2=search_engine2)