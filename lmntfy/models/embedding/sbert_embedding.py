import os
from sentence_transformers import SentenceTransformer
from . import Embedding

# needed to avoid a deadlock when processing sentences
os.environ["TOKENIZERS_PARALLELISM"]="False"

class SBERTEmbedding(Embedding):
    def __init__(self, 
                 name='all-mpnet-base-v2', 
                 embedding_length=768,
                 tokenizer=None,
                 max_input_tokens=384,
                 normalized=True):
        super().__init__(name, embedding_length, tokenizer, max_input_tokens, normalized)
        self.model = SentenceTransformer(name)

    def _embed(self, text):
        """
        OpenAI specific embedding computation.
        """
        text = text.replace("\n", " ")
        return self.model.encode([text])[0].tolist()

