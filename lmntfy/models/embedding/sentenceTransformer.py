import os
from sentence_transformers import SentenceTransformer
from . import Embedding

#--------------------------------------------------------------------------------------------------
# ABSTRACT CLASS

# needed to avoid a deadlock when processing sentences
os.environ["TOKENIZERS_PARALLELISM"]="False"

class SentenceTransformerEmbedding(Embedding):
    """
    Class for SBert models
    See this page for a good list: https://www.sbert.net/docs/pretrained_models.html
    """
    def __init__(self, 
                 models_folder,
                 name='all-mpnet-base-v2', 
                 embedding_length=768,
                 context_size=384,
                 normalized=True,
                 query_prefix='',
                 passage_prefix='',
                 device='cuda'):
        super().__init__(models_folder, name, embedding_length, context_size, normalized,
                         query_prefix, passage_prefix, device)
        # loads the model
        self.model = SentenceTransformer(self.pretrained_model_name_or_path, device=device)

    def _embed(self, text, is_query=False):
        """
        SBERT specific embedding computation.
        """
        return self.model.encode([text], convert_to_numpy=True, normalize_embeddings=self.normalized)[0]

#--------------------------------------------------------------------------------------------------
# MODELS

class MPNetEmbedding(SentenceTransformerEmbedding):
    """Default (generalist) SBert embeddings"""
    def __init__(self, 
                 models_folder,
                 name='all-mpnet-base-v2', 
                 embedding_length=768,
                 context_size=384,
                 normalized=True,
                 device=None):
        super().__init__(models_folder, name, embedding_length, context_size, normalized, device=device)

class E5LargeEmbedding(SentenceTransformerEmbedding):
    """
    https://huggingface.co/intfloat/e5-large-v2
    """
    def __init__(self, 
                 models_folder,
                 name='e5-large-v2', 
                 embedding_length=1024,
                 context_size=512,
                 normalized=True,
                 query_prefix="query: ",
                 passage_prefix="passage: ",
                 device='cuda'):
        super().__init__(models_folder, name, embedding_length, context_size, normalized, 
                         query_prefix, passage_prefix, device)

class E5BaseEmbedding(SentenceTransformerEmbedding):
    """
    https://huggingface.co/intfloat/e5-base-v2
    """
    def __init__(self, 
                 models_folder,
                 name='e5-base-v2', 
                 embedding_length=768,
                 context_size=512,
                 normalized=True,
                 query_prefix="query: ",
                 passage_prefix="passage: ",
                 device='cuda'):
        super().__init__(models_folder, name, embedding_length, context_size, normalized, 
                         query_prefix, passage_prefix, device)

class GISTEmbedding(SentenceTransformerEmbedding):
    """
    https://huggingface.co/avsolatorio/GIST-large-Embedding-v0
    """
    def __init__(self, 
                 models_folder,
                 name='GIST-large-Embedding-v0', 
                 embedding_length=1024,
                 context_size=512,
                 normalized=True,
                 query_prefix='',
                 passage_prefix='',
                 device='cuda'):
        super().__init__(models_folder, name, embedding_length, context_size, normalized, 
                         query_prefix, passage_prefix, device)

class BGELargeEmbedding(SentenceTransformerEmbedding):
    """
    https://huggingface.co/BAAI/bge-large-en-v1.5
    """
    def __init__(self, 
                 models_folder,
                 name='bge-large-en-v1.5', 
                 embedding_length=1024,
                 context_size=512,
                 normalized=True,
                 query_prefix='Represent this sentence for searching relevant passages: ',
                 passage_prefix='',
                 device='cuda'):
        super().__init__(models_folder, name, embedding_length, context_size, normalized, 
                         query_prefix, passage_prefix, device)

class NomicEmbedding(SentenceTransformerEmbedding):
    """
    https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
    """
    def __init__(self, 
                 models_folder,
                 name='nomic-embed-text-v1.5', 
                 embedding_length=768,
                 context_size=8192,
                 normalized=True,
                 query_prefix='search_query: ',
                 passage_prefix='search_document: ',
                 device='cuda'):
        Embedding.__init__(self, models_folder, name, embedding_length, context_size, normalized,
                           query_prefix, passage_prefix, device)
        # loads the model
        self.model = SentenceTransformer(self.pretrained_model_name_or_path, trust_remote_code=True, device=device)

class GTELargeEmbedding(SentenceTransformerEmbedding):
    """
    https://huggingface.co/thenlper/gte-large
    """
    def __init__(self, 
                 models_folder,
                 name='gte-large', 
                 embedding_length=1024,
                 context_size=512,
                 normalized=True,
                 query_prefix='',
                 passage_prefix='',
                 device='cuda'):
        super().__init__(models_folder, name, embedding_length, context_size, normalized, 
                         query_prefix, passage_prefix, device)