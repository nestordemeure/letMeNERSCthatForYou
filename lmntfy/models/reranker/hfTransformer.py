import torch
from pathlib import Path
from typing import List
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from . import Reranker
from ...database.document_splitter import markdown_splitter

#------------------------------------------------------------------------------
# ABSTRACT CLASS

class HFReranker(Reranker):
    """Rerankers based on hugginface transformers"""
    def __init__(self, models_folder:Path, name:str, device:str='cuda', context_length=512):
        super().__init__(models_folder, name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.pretrained_model_name_or_path, torch_dtype=torch.bfloat16).to(device)
        self.context_length = context_length

    def _count_tokens(self, text:str) -> int:
        """
        Counts the number of tokens used to represent the given text
        """
        tokens = self.tokenizer.encode(text, return_tensors='pt')
        token_number = tokens.size(-1)
        return token_number

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
        with torch.no_grad():
            tokens = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=self.context_length)
            for k, v in tokens.items():
                tokens[k] = v.to(self.device)
            scores = self.model(**tokens, return_dict=True).logits.view(-1, ).float()
            # gets the maximum similarity found
            similarity = scores.max().item()
        return similarity

#------------------------------------------------------------------------------
# MODELS

class BGELargeReranker(HFReranker):
    """https://huggingface.co/BAAI/bge-reranker-large"""
    def __init__(self, models_folder:Path, name:str='bge-reranker-large', device:str='cuda', context_length=512):
        super().__init__(models_folder, name, device, context_length)

class BGEBaseReranker(HFReranker):
    """https://huggingface.co/BAAI/bge-reranker-base"""
    def __init__(self, models_folder:Path, name:str='bge-reranker-base', device:str='cuda', context_length=512):
        super().__init__(models_folder, name, device, context_length)

class BCEBaseReranker(HFReranker):
    """https://huggingface.co/maidalun1020/bce-reranker-base_v1"""
    def __init__(self, models_folder:Path, name:str='bce-reranker-base_v1', device:str='cuda', context_length=512):
        super().__init__(models_folder, name, device, context_length)

class MXbaiLargeReranker(HFReranker):
    """https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1"""
    def __init__(self, models_folder:Path, name:str='mxbai-rerank-large-v1', device:str='cuda', context_length=512):
        super().__init__(models_folder, name, device, context_length)

class PRMBReranker(HFReranker):
    """https://huggingface.co/amberoad/bert-multilingual-passage-reranking-msmarco"""
    def __init__(self, models_folder:Path, name:str='bert-multilingual-passage-reranking-msmarco', device:str='cuda', context_length=512):
        super().__init__(models_folder, name, device, context_length)

class SimLMReranker(HFReranker):
    """https://huggingface.co/intfloat/simlm-msmarco-reranker"""
    def __init__(self, models_folder:Path, name:str='simlm-msmarco-reranker', device:str='cuda', context_length=192):
        super().__init__(models_folder, name, device, context_length)

    def _generate_all_pairs(self, query:str, passage:str, title='-') -> List[List[str]]:
        pairs = super()._generate_all_pairs(query, passage)
        pairs_formatted = [[query, f"{title}: {passage}"] for [query,passage] in pairs]
        return pairs