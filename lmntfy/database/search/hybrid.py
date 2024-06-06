import statistics
from functools import partial
from pathlib import Path
from typing import List, Tuple, Callable, Dict
from ..chunk import Chunk
from . import SearchEngine

#----------------------------------------------------------------------------------------
# ORDERING

# basic addition function
addition = lambda x, y: x+y

def merge_and_sort_scores(scored_chunks: List[Tuple[float, int]], merging_strategy: Callable[[float, float], float] = addition) -> List[Tuple[float, int]]:
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

def assert_order(scored_chunks: List[Tuple[float, int]]):
    """
    Takes a list of (score, chunk_id) tuples.
    Throws a runtime error if the scores are not ordered (either increasing or decreasing).
    """
    if len(scored_chunks) == 0:
        return  # An empty list is considered ordered
    elif all(scored_chunks[i][0] >= scored_chunks[i + 1][0] for i in range(len(scored_chunks) - 1)):
        return # decreasing
    elif all(scored_chunks[i][0] <= scored_chunks[i + 1][0] for i in range(len(scored_chunks) - 1)):
        return # increasing
    else:
        raise RuntimeError("Scores are not ordered either in increasing or decreasing order.")

#----------------------------------------------------------------------------------------
# SCORING

def reciprocal_rank_scores(scored_chunks: List[Tuple[float, int]], ranking_constant:int=60) -> List[Tuple[float, int]]:
    """
    Takes a list of (score,chunk_id) in order.
    And replace their scores with reciprocal ranks: score = 1 / (ranking_constant + rank)

    This is a solid default.
    ranking_constant defaults to 60, it is not expected to need any tuning

    see:
    * https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking
    * https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf
    """
    # asserts the ordering of the items
    assert_order(scored_chunks)
    # computes the rank based scores
    def reciprocal_rank(rank):
        return 1 / (ranking_constant + rank)
    return [(reciprocal_rank(rank),chunk_id) for (rank,(score,chunk_id)) in enumerate(scored_chunks, start=1)]

def relative_scores(scored_chunks: List[Tuple[float, int]]) -> List[Tuple[float, int]]:
    """
    Takes a list of (score,chunk_id) in order.
    And normalise them: score = (score - min(scores)) / (max(scores) - min(scores))

    This can be sensitive to score distributions.
    But is compatible with [autocut](https://docsbot.ai/article/advanced-rag-trim-the-irrelevant-context-using-autocut) type of systems.

    see: https://weaviate.io/blog/hybrid-search-fusion-algorithms#relativescorefusion
    """
    # asserts the ordering of the items
    assert_order(scored_chunks)
    # normalise the scores
    max_score = scored_chunks[0][0]
    min_score = scored_chunks[-1][0]
    def normalize(score):
        return (score - min_score) / (max_score - min_score)
    return [(normalize(score),chunk_id) for (score,chunk_id) in scored_chunks]

def distribution_based_scores(scored_chunks: List[Tuple[float, int]]) -> List[Tuple[float, int]]:
    """
    Takes a list of (score,chunk_id) in order.
    And normalize them: score = (score - (mean - 3std)) / ((mean + 3std) - (mean - 3std))

    This tries to improve over relative scores fusion by taking the distribution into account.

    see: https://medium.com/plain-simple-software/distribution-based-score-fusion-dbsf-a-new-approach-to-vector-search-ranking-f87c37488b18
    """
    # asserts the ordering of the items
    assert_order(scored_chunks)   
    # Calculate mean and standard deviation
    scores = [score for (score, chunk_id) in scored_chunks]
    mean_score = statistics.mean(scores)
    std_dev = statistics.stdev(scores)
    # Calculate the lower and upper bounds for normalization
    upper_bound = mean_score + 3 * std_dev
    lower_bound = mean_score - 3 * std_dev
    # flips bound when dealing with increasing scores
    max_score = scored_chunks[0][0]
    min_score = scored_chunks[-1][0]
    if max_score < min_score:
        lower_bound, upper_bound = upper_bound, lower_bound
    # Normalize the scores
    def normalize(score):
        return (score - lower_bound) / (upper_bound - lower_bound)
    return [(normalize(score), chunk_id) for score, chunk_id in scored_chunks]
    
#----------------------------------------------------------------------------------------
# SEARCH ENGINE

class HybridSearch_raw(SearchEngine):
    """
    Hybird search (also called Semantic search in traditional search engines).
    Combining the result of two (could be more) search engines, usualy vector search and keyword search.
    
    See:
    * <https://www.assembled.com/blog/better-rag-results-with-reciprocal-rank-fusion-and-hybrid-search>
    * <https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/azure-ai-search-outperforming-vector-search-with-hybrid/ba-p/3929167>
    * <https://weaviate.io/blog/hybrid-search-fusion-algorithms>
    """
    def __init__(self, search_engine1: SearchEngine, search_engine2: SearchEngine, 
                 scoring_function=reciprocal_rank_scores,
                 name:str='hybrid'):
        # search engines we are augmenting
        self.search_engine1: SearchEngine = search_engine1
        self.search_engine2: SearchEngine = search_engine2
        # hybridization functions
        self.scoring_function = scoring_function
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
    
    def get_closest_chunks(self, input_text: str, chunks:Dict[int,Chunk], k: int) -> List[Tuple[float,int]]:
        """
        Returns the (score,chunk_id) of the closest chunks, from best to worst
        """
        # gets the original results
        scored_chunks1 = self.search_engine1.get_closest_chunks(input_text, k)
        scored_chunks2 = self.search_engine2.get_closest_chunks(input_text, k)
        # rescores them
        rescored_chunks1 = self.scoring_function(scored_chunks1)
        rescored_chunks2 = self.scoring_function(scored_chunks2)
        # merges both
        rescored_chunks = rescored_chunks1 + rescored_chunks2
        # sort the chunks according to the new score
        rescored_chunks = merge_and_sort_scores(rescored_chunks, merging_strategy=addition)
        return rescored_chunks

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
HybridSearch = lambda search_engine1, search_engine2, scoring_function: partial(HybridSearch_raw, search_engine1=search_engine1, search_engine2=search_engine2, scoring_function=scoring_function)