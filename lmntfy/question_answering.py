from typing import List, Dict
from .models import LanguageModel, Embedding
from .database import Database
from pathlib import Path
from datetime import datetime
import traceback
import json


class QuestionAnswerer:
    def __init__(self, llm:LanguageModel, embeder:Embedding, database:Database):
        self.llm = llm
        self.embeder = embeder
        self.database = database
        # stored for debugging purposes
        self.latest_question = None
        self.latest_chunks = None

    async def get_answer(self, question:str, max_context_size=8, verbose=False) -> str:
        """gets the string answer to a single question"""
        # build a single message discussion
        messages = [{'role': 'user', 'content': question}]
        # extracts the keywords
        keywords = await self.llm.extract_question(messages, verbose=verbose)
        # get a context to help us answer the question
        chunks = self.database.get_closest_chunks(keywords, max_context_size)
        # gets an answer from the model
        answer = await self.llm.chat(messages, chunks, verbose=verbose)
        # stores information for later debugging
        self.latest_question = question
        self.latest_chunks = chunks
        return answer

    async def chat(self, messages:List[Dict], max_context_size=8, verbose=False) -> Dict:
        """
        Answers the latest question given a list of messages.
        Message are expectted to be dictionnaries with a 'role' ('user' or 'assistant') and 'content' field.
        the question returned will be a message with role 'assistant'.
        """
        # extracts the keywords
        keywords = await self.llm.extract_question(messages, verbose=verbose)
        # get a context to help us answer the question
        chunks = self.database.get_closest_chunks(keywords, max_context_size)
        # gets an answer from the model
        answer = await self.llm.chat(messages, chunks, verbose=verbose)
        # stores information for later debugging
        self.latest_question = keywords
        self.latest_chunks = chunks
        return {'role':'assistant', 'content': answer}
