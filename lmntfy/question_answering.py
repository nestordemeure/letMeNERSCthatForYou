from .models.llm import LanguageModel
from .models.embedding import Embedding
from .database import Database

class QuestionAnswerer:
    def __init__(self, llm:LanguageModel, embeder:Embedding, database:Database, path=None):
        self.llm = llm
        self.embeder = embeder
        self.database = database

    def get_answer(self, question, max_context_size=5, verbose=False):
        # get a context to help us answer the question
        chunks = self.database.get_closest_chunks(question, max_context_size)
        # gets an answer from the model
        answer = self.llm.get_answer(question, chunks, verbose=verbose)
        return answer