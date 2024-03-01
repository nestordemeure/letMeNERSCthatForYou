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
                 max_input_tokens=384,
                 normalized=True,
                 query_prefix='',
                 passage_prefix='',
                 device='cuda'):
        super().__init__(models_folder, name, embedding_length, max_input_tokens, normalized,
                         query_prefix, passage_prefix, device)
        # ensures that the model caching folder is set properly
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(models_folder)
        # loads the model
        self.model = SentenceTransformer(self.pretrained_model_name_or_path, device=device)

    def _embed(self, text, is_query=False):
        """
        SBERT specific embedding computation.
        """
        return self.model.encode([text], convert_to_numpy=True, normalize_embeddings=self.normalized)[0]

    def count_tokens(self, text):
        """
        Counts the number of tokens used to represent the given text
        """
        encoded_text = self.model.tokenize([text])['input_ids']
        return encoded_text.numel()

#--------------------------------------------------------------------------------------------------
# MODELS

class MPNetEmbedding(SentenceTransformerEmbedding):
    """Default (generalist) SBert embeddings"""
    def __init__(self, 
                 models_folder,
                 name='all-mpnet-base-v2', 
                 embedding_length=768,
                 max_input_tokens=384,
                 normalized=True,
                 device=None):
        super().__init__(models_folder, name, embedding_length, max_input_tokens, normalized, device=device)

class QAMPNetEmbedding(SentenceTransformerEmbedding):
    """Q&A-tuned SBert embeddings"""
    def __init__(self, 
                 models_folder,
                 name='multi-qa-mpnet-base-cos-v1', 
                 embedding_length=768,
                 max_input_tokens=384,
                 normalized=True,
                 device=None):
        super().__init__(models_folder, name, embedding_length, max_input_tokens, normalized, device=device)

class SOMPNetEmbedding(SentenceTransformerEmbedding):
    """
    Stackoverflow tuned SBert embeddings
    https://huggingface.co/flax-sentence-embeddings/stackoverflow_mpnet-base
    """
    def __init__(self, 
                 models_folder,
                 name='stackoverflow_mpnet-base', 
                 embedding_length=768,
                 max_input_tokens=384,
                 normalized=True,
                 device=None):
        super().__init__(models_folder, name, embedding_length, max_input_tokens, normalized, device=device)

class E5LargeEmbedding(SentenceTransformerEmbedding):
    """
    https://huggingface.co/intfloat/e5-large-v2
    """
    def __init__(self, 
                 models_folder,
                 name='e5-large-v2', 
                 embedding_length=1024,
                 max_input_tokens=512,
                 normalized=True,
                 query_prefix="query: ",
                 passage_prefix="passage: ",
                 device='cuda'):
        super().__init__(models_folder, name, embedding_length, max_input_tokens, normalized, 
                         query_prefix, passage_prefix, device)

class E5BaseEmbedding(SentenceTransformerEmbedding):
    """
    https://huggingface.co/intfloat/e5-base-v2
    """
    def __init__(self, 
                 models_folder,
                 name='e5-base-v2', 
                 embedding_length=768,
                 max_input_tokens=512,
                 normalized=True,
                 query_prefix="query: ",
                 passage_prefix="passage: ",
                 device='cuda'):
        super().__init__(models_folder, name, embedding_length, max_input_tokens, normalized, 
                         query_prefix, passage_prefix, device)
