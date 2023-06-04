import openai
from . import Embedding
from .. import retry

class OpenAIEmbedding(Embedding):
    def __init__(self, 
                 name='text-embedding-ada-002', 
                 embedding_length=1536,
                 tokenizer='cl100k_base',
                 max_input_tokens=8191,
                 normalized=True):
        super().__init__(name, embedding_length, tokenizer, max_input_tokens, normalized)

    @retry(n=5)
    def _embed(self, text):
        """
        OpenAI specific embedding computation.
        """
        text = text.replace("\n", " ")
        return openai.Embedding.create(input=[text], model=self.name)['data'][0]['embedding']
