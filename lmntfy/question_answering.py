from typing import List, Dict
from .models import LanguageModel, Embedding
from .database import Database
from pathlib import Path
from datetime import datetime
import traceback
import json

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
            # extract the latest question
            question = self.llm.extract_question(messages, verbose=verbose)
            print(f"QUESTION: '{question}'")
            # gets an answer for the question
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