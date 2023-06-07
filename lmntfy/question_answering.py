from .models.llm import LanguageModel
from .models.embedding import Embedding
from .database import Database

class QuestionAnswerer:
    def __init__(self, llm:LanguageModel, embeder:Embedding, database:Database, path=None):
        self.llm = llm
        self.embeder = embeder
        self.database = database

    def get_answer(self, question, max_context_size=8, verbose=False):
        # get a context to help us answer the question
        chunks = self.database.get_closest_chunks(question, max_context_size)
        # gets an answer from the model
        answer = self.llm.get_answer(question, chunks, verbose=verbose)
        return answer

    def continue_chat(self, messages, max_context_size=8, verbose=False):
        """
        Answers the latest question given a list of messages.
        Message are expectted to be dictionnaries with a 'role' ('user' or 'assistant') and 'content' field.
        the question returned will be a message with role 'assistant'.
        """
        # extract the latest question
        question = self.llm.extract_question(messages, verbose=verbose)
        # gets an answer for the question
        answer = self.get_answer(question, max_context_size=max_context_size, verbose=verbose)
        return {'role':'assistant', 'content': answer}