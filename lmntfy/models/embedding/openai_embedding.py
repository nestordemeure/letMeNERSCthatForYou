import openai
import tiktoken
import numpy as np
from . import Embedding
from .. import retry

class OpenAIEmbedding(Embedding):
    def __init__(self, 
                 name='text-embedding-ada-002', 
                 embedding_length=1536,
                 max_input_tokens=8191,
                 normalized=True):
        super().__init__(name, embedding_length, max_input_tokens, normalized)
        self.tokenizer = tiktoken.get_encoding('cl100k_base')

    @retry(n=5)
    def _embed(self, text):
        """
        OpenAI specific embedding computation.
        """
        text = text.replace("\n", " ")
        embedding_list = openai.Embedding.create(input=[text], model=self.name)['data'][0]['embedding']
        return np.array(embedding_list, dtype='float32')

    def token_counter(self, text):
        """
        Counts the number of tokens used to represent the given text
        """
        encoded_text = self.tokenizer.encode(text)
        return len(encoded_text)