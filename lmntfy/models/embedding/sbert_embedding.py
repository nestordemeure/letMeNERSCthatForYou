from sentence_transformers import SentenceTransformer
from . import Embedding
from .. import retry

class SBERTEmbedding(Embedding):
    def __init__(self, 
                 name='all-mpnet-base-v2', 
                 embedding_length=768,
                 tokenizer='cl100k_base',
                 max_input_tokens=384,
                 normalized=True):
        super().__init__(name, embedding_length, tokenizer, max_input_tokens, normalized)
        self.model = SentenceTransformer(name)

    @retry(n=5)
    def _embed(self, text):
        """
        OpenAI specific embedding computation.
        """
        text = text.replace("\n", " ")
        return self.model.encode([text])[0].tolist()

