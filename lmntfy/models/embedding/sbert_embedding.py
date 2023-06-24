import os
from sentence_transformers import SentenceTransformer
from . import Embedding

# needed to avoid a deadlock when processing sentences
os.environ["TOKENIZERS_PARALLELISM"]="False"

class SBERTEmbedding(Embedding):
    def __init__(self, 
                 models_folder,
                 name='all-mpnet-base-v2', 
                 embedding_length=768,
                 max_input_tokens=384,
                 normalized=True):
        super().__init__(models_folder, name, embedding_length, max_input_tokens, normalized)
        # ensures that the model caching folder is set properly
        os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(models_folder)
        # loads the model
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
        encoded_text = self.model.tokenize([text])['input_ids']
        return encoded_text.numel()