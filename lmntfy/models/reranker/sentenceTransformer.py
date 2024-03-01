import numpy as np
from pathlib import Path
from typing import List
from . import Reranker
from sentence_transformers import CrossEncoder
from ...database.document_loader.markdown_spliter import markdown_splitter

#------------------------------------------------------------------------------
# ABSTRACT CLASS

class STReranker(Reranker):
    """
    Rerankers based on sentence transformers
    https://www.sbert.net/examples/applications/cross-encoder/README.html
    """
    def __init__(self, models_folder:Path, name:str, device:str='cuda', context_length=512):
        super().__init__(models_folder, name, device)
        self.model = CrossEncoder(self.pretrained_model_name_or_path, device=device)
        self.context_length = context_length

    def _count_tokens(self, text:str) -> int:
        """
        Counts the number of tokens used to represent the given text
        """
        encoded_text = self.model.tokenize([text])['input_ids']
        return encoded_text.numel()

    def _generate_all_pairs(self, query:str, passage:str) -> List[List[str]]:
        """
        takes a query and a passage
        split them if they are too long for our context length
        returns all possible pairs of query / passage
        """
        # turns query into pairs if needed
        query_list = [query]
        if self._count_tokens(query) > self.context_length:
            sub_chunks = markdown_splitter('', query, self._count_tokens, self.context_length)
            query_list = [chunk.content for chunk in sub_chunks]
        # turn passage into pairs if needed
        passage_list = [passage]
        if self._count_tokens(passage) > self.context_length:
            sub_chunks = markdown_splitter('', passage, self._count_tokens, self.context_length)
            passage_list = [chunk.content for chunk in sub_chunks]
        # build all pairs of query*passage
        result = []
        for query in query_list:
            for passage in passage_list:
                result.append([query,passage])
        return result

    def _similarity(self, query:str, passage:str) -> float:
        """
        Compute the similarity between a query and a passage
        """
        # generate text pairs to be evaluated
        pairs = self._generate_all_pairs(query, passage)
        # measures the similarity between each text pair
        scores = self.model.predict(pairs, convert_to_numpy=True)
        # gets the maximum similarity found
        similarity = np.max(scores)
        return similarity

#------------------------------------------------------------------------------
# MODELS

class MXbaiLargeReranker(STReranker):
    """https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1"""
    def __init__(self, models_folder:Path, name:str='mxbai-rerank-large-v1', device:str='cuda', context_length=512):
        super().__init__(models_folder, name, device, context_length)
