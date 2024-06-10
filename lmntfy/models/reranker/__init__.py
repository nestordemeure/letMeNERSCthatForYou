from abc import ABC, abstractmethod
from typing import List
from pathlib import Path
from ...database.document_splitter import Chunk

class Reranker(ABC):
    """
    Used to compute similarity between a query and passages
    See this page for a comparison of various rerankers: https://huggingface.co/spaces/mteb/leaderboard
    """
    def __init__(self, models_folder:Path, name:str, device:str='cuda'):
        self.name = name
        self.pretrained_model_name_or_path = str(models_folder / name)
        self.device = device
    
    @abstractmethod
    def _similarity(self, query:str, passage:str) -> float:
        """
        Abstract method to compute the similarity between a query and a passage
        """
        pass

    def similarity(self, query:str, passage:str | Chunk) -> float:
        """
        Compute the similarity between a query and a passage
        """
        # extracts the text if need be
        passage = passage.content if isinstance(passage,Chunk) else passage
        # TODO if a passage is too long then slice it, compute all similarities and keep the maximum
        return self._similarity(query, passage)

    def similarities(self, query:str, passages:List[str | Chunk]) -> List[float]:
        """
        Produces a list of similarities for given passages.
        """
        return [self.similarity(query, passage) for passage in passages]

    def rerank(self, query:str, passages:List[str | Chunk], return_similarities=False) -> List[str | Chunk]:
        """
        Takes various passages and re-sorts them by similarity to the query (from high to low).
        """
        # computes the similarities
        similarities = self.similarities(query, passages)
        # sorts the passages according to the similarities
        passages_similarities = list(zip(passages, similarities))
        sorted_passages_similarities = sorted(passages_similarities, key=lambda x: x[1], reverse=True)
        # returns
        if return_similarities:
            return sorted_passages_similarities
        else:
            sorted_passages = [passage for (passage,similarity) in sorted_passages_similarities]
            return sorted_passages
    
    def keep_most_similar(self, query:str, passages:List[str | Chunk]) -> List[str | Chunk]:
        """
        Given a query and a list of passages, returns only the passages deemed similar enough to the query
        assumes the least similar is a NO and the most similar is a YES
        then, for all passages, puts them in the cluster whose current mean is closest to them
        """
        # shortcut if there are not enough passages
        if len(passages) < 2: return passages
        # computes the similarities and sorts the passages from best to worst
        passages_similarities = self.rerank(query, passages)
        # utility function
        def mean(passages_similarities) -> float:
            total_similarity = sum(similarity for (passage,similarity) in passages_similarities)
            return total_similarity / len(passages_similarities)
        # best
        index_last_kept = 0
        passages_kept = [passages_similarities[index_last_kept]]
        mean_similarity_kept = mean(passages_kept)
        # worst
        index_last_discarded = len(passages_similarities)-1
        passages_discarded = [passages_similarities[index_last_discarded]]
        mean_similarity_discarded = mean(passages_discarded)
        # process all passages
        while (index_last_kept+1 < index_last_discarded):
            # potential kept
            passage_to_keep, similarity_to_keep = passages_similarities[index_last_kept+1]
            keeping_distance = mean_similarity_kept - similarity_to_keep
            # potential discard
            passage_to_discard, similarity_to_discard = passages_similarities[index_last_discarded-1]
            discarding_distance = similarity_to_discard - mean_similarity_discarded
            # should we keep or discard?
            if (keeping_distance < discarding_distance):
                index_last_kept += 1
                passages_kept.append( (passage_to_keep, similarity_to_keep) )
                mean_similarity_kept = mean(passages_kept)
            else:
                index_last_discarded -= 1
                passages_discarded.append( (passage_to_discard, similarity_to_discard) )
                mean_similarity_discarded = mean(passages_discarded)
        # remove similarity information
        passages_kept = [passage for (passage,similarity) in passages_kept]
        return passages_kept

from .noop import NoReranker # does nothing, triggers on call
from .tfidf import TFIDFReranker # keyword based: can miss but very orthogonal to classic sentence embedding
from .hfTransformer import BGEBaseReranker # a bit weaker than BGE large
from .hfTransformer import BGELargeReranker # a bit lower than tfidf
from .hfTransformer import BCEBaseReranker # a bit weaker than BGE large
from .hfTransformer import MXbaiLargeReranker # a bit above BGE large?
from .hfTransformer import PRMBReranker # very weak, best avoided
from .hfTransformer import SimLMReranker # great on some and weak on others
# default reranker
Default = TFIDFReranker