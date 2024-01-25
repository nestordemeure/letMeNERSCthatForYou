from typing import List, Dict
from enum import Enum, auto
from .models import LanguageModel, Embedding
from .database import Database
from pathlib import Path
from datetime import datetime
import traceback
import json

#----------------------------------------------------------------------------------------
# ANSWER TYPE

class AnswerType(Enum):
    OUT_OF_SCOPE = auto()
    QUESTION = auto()
    SMALL_TALK = auto()

class Answer:
    """output of the triage operation"""
    def __init__(self, answer_type: AnswerType, content: str = None, raw:str = None):
        self.answer_type = answer_type
        self.content = content
        self.raw = raw

    @classmethod
    def out_of_scope(cls, raw:str=None):
        return cls(AnswerType.OUT_OF_SCOPE, raw=raw)

    @classmethod
    def question(cls, question: str, raw:str=None):
        return cls(AnswerType.QUESTION, content=question, raw=raw)

    @classmethod
    def smallTalk(cls, answer: str, raw:str=None):
        return cls(AnswerType.SMALL_TALK, content=answer, raw=raw)

    def is_out_of_scope(self):
        return self.answer_type == AnswerType.OUT_OF_SCOPE

    def is_question(self):
        return self.answer_type == AnswerType.QUESTION

    def is_smallTalk(self):
        return self.answer_type == AnswerType.SMALL_TALK

    def __str__(self):
        if self.is_out_of_scope():
            return "OUT_OF_SCOPE"
        elif self.is_question():
            return f"QUESTION({self.content})"
        elif self.is_smallTalk():
            return f"SMALL_TALK({self.content})"
        else:
            return "Invalid Answer Type"

# answer returned if the model decides the question is out of scope
out_of_scope_answer = """
It seems your inquiry is outside the scope of this documentation chatbot.\
We focus on providing assistance within the scope of NERSC's documentation.\
For general or unrelated queries, we recommend consulting appropriate resources such as NERSC support or experts in that field.\
If you have any NERSC-related questions, feel free to ask!

References:
* <https://www.nersc.gov/>
* <https://docs.nersc.gov/>
* <https://www.nersc.gov/users/getting-help/online-help-desk/>
"""

#----------------------------------------------------------------------------------------
# MODEL CALLS

class QuestionAnswerer:
    def __init__(self, llm:LanguageModel, embeder:Embedding, database:Database, logs_folder:Path=None):
        self.llm = llm
        self.embeder = embeder
        self.database = database
        self.logs_folder = logs_folder

    def get_answer(self, question:str, max_context_size=8, verbose=False) -> str:
        try:
            # get a context to help us answer the question
            chunks = self.database.get_closest_chunks(question, max_context_size)
            # gets an answer from the model
            answer = self.llm.get_answer(question, chunks, verbose=verbose)
        except Exception as e:
            # propagate the exeption as usual
            if self.logs_folder is None:
                raise
            # returns the error message instead of the model's answer
            self.log_error(question, e)
            answer = f"ERROR: {str(e)}"
        return answer

    def continue_chat(self, messages:List[Dict], max_context_size=8, verbose=False) -> Dict:
        """
        Answers the latest question given a list of messages.
        Message are expectted to be dictionnaries with a 'role' ('user' or 'assistant') and 'content' field.
        the question returned will be a message with role 'assistant'.
        """
        try:
            # decide on what to do
            triage = self.llm.triage(messages, verbose=verbose)
            if triage.is_out_of_scope():
                # use prewritten out-of-scope answer
                answer = out_of_scope_answer
            elif triage.is_smallTalk():
                # use model's answer
                answer = triage.content
            else:
                # extract the latest question
                question = triage.content
                # gets an answer for the question using the documenation
                answer = self.get_answer(question, max_context_size=max_context_size, verbose=verbose)
        except Exception as e:
            # propagate the exeption as usual
            if self.logs_folder is None:
                raise
            # returns the error message instead of the model's answer
            self.log_error(messages, e)
            answer = f"ERROR: {str(e)}"
        return {'role':'assistant', 'content': answer}

    def log_error(self, input_data, error: BaseException):
        """Saves the input data and error message to a JSON log file in self.logs_folder, if it's not None."""
        if self.logs_folder is None: return
        current_time = datetime.now()
        log_file_path = self.logs_folder / f"{current_time.strftime('%Y-%m-%d_%H-%M-%S')}_error_log.json"
        with open(log_file_path, 'w') as f:
            log_entry = {'input': input_data, 'error_message': str(error), 'stacktrace': traceback.format_exc()}
            json.dump(log_entry, f, indent=2)