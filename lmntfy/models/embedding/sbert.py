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
                 device=None):
        super().__init__(models_folder, name, embedding_length, max_input_tokens, normalized, device)
        # ensures that the model caching folder is set properly
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(models_folder)
        # loads the model
        self.model = SentenceTransformer(name, device=device)

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
        super().__init__(models_folder, name, embedding_length, max_input_tokens, normalized, device)

class QAMPNetEmbedding(SentenceTransformerEmbedding):
    """Default SBert embeddings"""
    def __init__(self, 
                 models_folder,
                 name='multi-qa-mpnet-base-cos-v1', 
                 embedding_length=768,
                 max_input_tokens=384,
                 normalized=True,
                 device=None):
        super().__init__(models_folder, name, embedding_length, max_input_tokens, normalized, device)