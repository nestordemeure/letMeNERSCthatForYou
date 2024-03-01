from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np

class Embedding(ABC):
    """
    See this page for a comparison of various embeddings: https://huggingface.co/spaces/mteb/leaderboard
    """
    def __init__(self, models_folder:Path, name:str, 
                 embedding_length:int, max_input_tokens:int, normalized:bool, 
                 query_prefix:str='', passage_prefix:str='',
                 device:str='cuda'):
        """
        Parameters:
            models_folder (Path): The path to the directory containing the model files.
            name (str): The name of the embedding model.
            embedding_length (int): The size of the embedding vectors produced by the model.
            max_input_tokens (int): The maximum number of tokens that can be processed in one input sequence.
            normalized (bool): A flag indicating whether the model's output embeddings are normalized.
            query_prefix (str): optional prefix to put in front of queries
            passage_prefix (str): optional prefix to put in front of passages
            device (str): on which device should the model be

        """
        self.pretrained_model_name_or_path = str(models_folder / name)
        self.models_folder = models_folder
        self.name = name
        self.embedding_length = embedding_length
        self.max_input_tokens = max_input_tokens
        self.normalized = normalized
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.device=device

    def embed(self, text:str, is_query=False) -> np.ndarray:
        """
        Converts text into an embedding.
        """
        try:
            text = (self.query_prefix + text) if is_query else (self.passage_prefix + text)
            raw_embedding = self._embed(text, is_query)
        except Exception as e:
            print(f"An error occurred while embedding the text '{text}': {str(e)}")
            raise  # rethrow the exception after handling
        
        if self.normalized:
            return raw_embedding
        else:
            # normalize the embedding
            norm = np.linalg.norm(raw_embedding)
            if norm == 0: 
                return raw_embedding
            return raw_embedding / norm

    @abstractmethod
    def _embed(self, text:str, is_query=False) -> np.ndarray:
        """
        Abstract method for converting text into an embedding.
        is_query is there for models who have different methods to embed queries vs normal text
        (note that we ake prefixes into account above this level)
        """
        pass

    @abstractmethod
    def count_tokens(self, text:str) -> int:
        """
        Counts the number of tokens used to represent the given text
        """
        pass

from .sentenceTransformer import MPNetEmbedding # good overall default
from .sentenceTransformer import QAMPNetEmbedding # finetuned for Q&A, weaker than MPNet
from .sentenceTransformer import SOMPNetEmbedding # stackoverflow finetuned, weaker than MPNet
from .hfTransformer import UAEEmbedding # weaker than MPNet on our tests
from .sentenceTransformer import E5BaseEmbedding # a bit weaker than large
from .sentenceTransformer import E5LargeEmbedding # somewhat better than MPNet?
from .sentenceTransformer import GISTEmbedding # somewhat weaker than MPNet
from .sentenceTransformer import BGELargeEmbedding # somewhat better than E5Large
from .sentenceTransformer import NomicEmbedding # does fine, massive (useless) context size
from .sentenceTransformer import GTELargeEmbedding # a bit inferior to BGE large
# embeddings used by default everywhere
Default = BGELargeEmbedding
