import os
from sentence_transformers import SentenceTransformer
from . import Embedding

# needed to avoid a deadlock when processing sentences
os.environ["TOKENIZERS_PARALLELISM"]="False"

class SBERTEmbedding(Embedding):
    def __init__(self, 
                 name='all-mpnet-base-v2', 
                 embedding_length=768,
                 max_input_tokens=384,
                 normalized=True):
        super().__init__(name, embedding_length, max_input_tokens, normalized)
        self.model = SentenceTransformer(name)

    def _embed(self, text):
        """
        SBERT specific embedding computation.
        """
        text = text.replace("\n", " ")
        return self.model.encode([text], convert_to_numpy=True, normalize_embeddings=self.normalized)[0]

    def token_counter(self, text):
        """
        Counts the number of tokens used to represent the given text
        """
        encoded_text = self.model.tokenize([text])[0]
        return len(encoded_text)