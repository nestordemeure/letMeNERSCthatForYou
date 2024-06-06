import statistics
from pathlib import Path
from typing import List, Tuple, Callable, Dict
from ..chunk import Chunk
from . import SearchEngine

#----------------------------------------------------------------------------------------
# ORDERING

# basic addition function
addition = lambda x, y: x+y

def merge_and_sort_scores(scored_chunk_ids: List[Tuple[float, int]], merging_strategy: Callable[[float, float], float] = addition) -> List[Tuple[float, int]]:
    """
    Takes a list of (score, chunk_id) and:
    * merges identical chunks using the given merging strategy (addition, max, etc)
    * sorts them from largest to smallest by score
    """
    # Merge identical chunks using the given merging strategy
    chunk_dict = {}
    for score, chunk_id in scored_chunk_ids:
        if chunk_id in chunk_dict:
            chunk_dict[chunk_id] = merging_strategy(chunk_dict[chunk_id], score)
        else:
            chunk_dict[chunk_id] = score

    # Convert the dictionary back to a list of tuples
    merged_list = [(score, chunk_id) for chunk_id, score in chunk_dict.items()]

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
        raise RuntimeError("Scores are in INCREASING order.")
    else:
        raise RuntimeError("Scores are not ordered.")

#----------------------------------------------------------------------------------------
# SCORING

def reciprocal_rank_scores(scored_chunks: List[Tuple[float, int]], k:int, ranking_constant:int=60) -> List[Tuple[float, int]]:
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
    if len(scored_chunks) == 0: return scored_chunks
    assert_order(scored_chunks)
    # computes the rank based scores
    def reciprocal_rank(rank):
        return 1 / (ranking_constant + rank)
    return [(reciprocal_rank(rank),chunk_id) for (rank,(score,chunk_id)) in enumerate(scored_chunks, start=1)]

def relative_scores(scored_chunks: List[Tuple[float, int]], k:int) -> List[Tuple[float, int]]:
    """
    Takes a list of (score,chunk_id) in order.
    And normalise them: score = (score - min(scores)) / (max(scores) - min(scores))

    This can be sensitive to score distributions.
    But is compatible with [autocut](https://docsbot.ai/article/advanced-rag-trim-the-irrelevant-context-using-autocut) type of systems.

    see: https://weaviate.io/blog/hybrid-search-fusion-algorithms#relativescorefusion
    """
    # asserts the ordering of the items
    if len(scored_chunks) == 0: return scored_chunks
    assert_order(scored_chunks)
    # normalise the scores
    # NOTE: use only the best k items to avoid giving undue weight to a search engine returning more results
    k = min(k, len(scored_chunks))
    max_score = scored_chunks[0][0]
    min_score = scored_chunks[k-1][0]
    def normalize(score):
        return (score - min_score) / (max_score - min_score)
    return [(normalize(score),chunk_id) for (score,chunk_id) in scored_chunks]

def distribution_based_scores(scored_chunks: List[Tuple[float, int]], k:int) -> List[Tuple[float, int]]:
    """
    Takes a list of (score,chunk_id) in order.
    And normalize them: score = (score - (mean - 3std)) / ((mean + 3std) - (mean - 3std))

    This tries to improve over relative scores fusion by taking the distribution into account.

    see: https://medium.com/plain-simple-software/distribution-based-score-fusion-dbsf-a-new-approach-to-vector-search-ranking-f87c37488b18
    """
    # asserts the ordering of the items
    if len(scored_chunks) == 0: return scored_chunks
    assert_order(scored_chunks)   
    # Calculate mean and standard deviation
    # NOTE: use only the best k items to avoid giving undue weight to a search engine returning more results
    k = min(k, len(scored_chunks))
    scores = [score for (score, chunk_id) in scored_chunks[:k]]
    mean_score = statistics.mean(scores)
    std_dev = statistics.stdev(scores)
    # Calculate the lower and upper bounds for normalization
    upper_bound = mean_score + 3 * std_dev
    lower_bound = mean_score - 3 * std_dev
    # Normalize the scores
    def normalize(score):
        return (score - lower_bound) / (upper_bound - lower_bound)
    return [(normalize(score), chunk_id) for score, chunk_id in scored_chunks]
    
#----------------------------------------------------------------------------------------
# SEARCH ENGINE

class HybridSearch(SearchEngine):
    """
    Hybird search (also called Semantic search in traditional search engines).
    Combining the result of two (could be more) search engines, usualy vector search and keyword search.
    
    See:
    * https://www.assembled.com/blog/better-rag-results-with-reciprocal-rank-fusion-and-hybrid-search
    * https://techcommunity.microsoft.com/t5/ai-azure-ai-services-blog/azure-ai-search-outperforming-vector-search-with-hybrid/ba-p/3929167
    * https://weaviate.io/blog/hybrid-search-fusion-algorithms

    NOTE: we could add a weight to each search algorithm when merging.
    """
    def __init__(self, search_engine1: SearchEngine, search_engine2: SearchEngine, 
                 scoring_function=reciprocal_rank_scores):
        # search engines we are augmenting
        self.search_engine1: SearchEngine = search_engine1
        self.search_engine2: SearchEngine = search_engine2
        # hybridization functions
        self.scoring_function = scoring_function
        # init parent
        super().__init__(name=f"hybrid_{search_engine1.name}_{search_engine2.name}")

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
        scored_chunks1 = self.search_engine1.get_closest_chunks(input_text, chunks, k)
        scored_chunks2 = self.search_engine2.get_closest_chunks(input_text, chunks, k)
        # rescores them
        rescored_chunks1 = self.scoring_function(scored_chunks1, k)
        rescored_chunks2 = self.scoring_function(scored_chunks2, k)
        # merges both
        rescored_chunks = rescored_chunks1 + rescored_chunks2
        # sort the chunks according to the new score
        rescored_chunks = merge_and_sort_scores(rescored_chunks, merging_strategy=addition)
        return rescored_chunks

    def initialize(self, database_folder:Path):
        """
        Initialize the search engine if needed.
        """
        self.search_engine1.initialize(database_folder)
        self.search_engine2.initialize(database_folder)

    def exists(self, database_folder:Path) -> bool:
        """
        Returns True if an instance of the search engine is saved in the given folder.
        """
        return self.search_engine1.exists(database_folder) and self.search_engine2.exists(database_folder)

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
