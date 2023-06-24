import re
import json
import tiktoken
import openai
from . import LanguageModel, keep_references_only
from .. import retry

#----------------------------------------------------------------------------------------
# PROMPTS

# prompt to answer a question
ANSWERING_PROMPT="You are a member of the NERSC supercomputing center's support staff. \
Generate a comprehensive and informative answer for a given question solely based on the provided web Search Results (URL and Extract). \
You must only use information from the provided search results. \
Use an unbiased and journalistic tone. \
Combine search results together into a coherent answer. \
Only cite the most relevant results that answer the question accurately. \
Try and be careful not to go off-topics. \
End your answer with \"References:\" followed by a bullet list of the relevant urls."

# prompt to get references for an answer
REFERENCE_PROMPT="Produce a bullet list of the urls you found useful to answer the question, \
sorted from most relevant to least relevant. \
Try to keep the bullet list short, keeping *only* the relevant urls (there are rarely more than three relevant urls)."

# prompt to summarize a conversation into its latest question
QUESTION_EXTRACTION_PROMPT_SYSTEM="You are a question extraction system. \
You will be provided the last messages of a conversation between a user of the NERSC supercomputing center and an assistant from its support, ending on a question by the user. \
Your task is to return the user's last question."
QUESTION_EXTRACTION_PROMPT_USER="Return the user's last question, rephrasing it such that it can be understood without the rest of the conversation."

#----------------------------------------------------------------------------------------
# MODEL

class GPT35(LanguageModel):
    def __init__(self, 
                 models_folder,
                 model_name='gpt-3.5-turbo', 
                 context_size=4096):
        super().__init__(models_folder, model_name, context_size)
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.model_tokens_per_message = 4
        self.model_tokens_per_name = -1
        self.upper_answer_size = 250
        self.upper_question_size = 200

    def token_counter_messages(self, messages):
        """
        GPT-3.5-turbo specific token counting implementation for list of messages.
        """
        total_tokens = 0
        for message in messages:
            total_tokens += self.model_tokens_per_message
            for key, value in message.items():
                total_tokens += self.token_counter(value)
                if key == "name":
                    total_tokens += self.model_tokens_per_name
        total_tokens += 2
        return total_tokens

    def token_counter(self, text):
        """
        GPT-3.5-turbo specific token counting implementation.
        """
        encoded_text = self.tokenizer.encode(text)
        return len(encoded_text)

    @retry(n=5)
    def query(self, messages, verbose=False):
        """
        GPT-3.5-turbo specific model query and response.
        """
        response = openai.ChatCompletion.create(model=self.model_name, messages=messages, temperature=0)
        if verbose: 
            full_messages = messages + [response.choices[0].message]
            text_messages = json.dumps(full_messages, indent=4)
            print(text_messages)
        return response.choices[0].message['content']

    def get_answer(self, question, chunks, verbose=False):
        """
        Method to get an answer given a question and some chunks passed for context.
        """
        # builds the prompt
        system_message = {"role": "system", "content": ANSWERING_PROMPT}
        context_messages = [{"role": "assistant", "content": str(chunk)} for chunk in chunks]
        question_message = {"role": "user", "content": question}
        messages = [system_message] + context_messages + [question_message]
        # keep as many context messages as we can
        while self.token_counter_messages(messages) + self.upper_answer_size > self.context_size:
            if len(messages) > 2:
                # reduce the context size by popping the latest context message
                # (which is likely the least relevant)
                messages.pop(-2)
                chunks = chunks[:-1]
            else:
                # no more space to reduce context size
                raise ValueError("You query is too long for the model's context size.")
        # runs the query
        answer = self.query(messages, verbose=verbose)
        # adds sources at the end of the query
        if not "References:" in answer:
            answer = self.add_references(question, answer, chunks, verbose=verbose)
        # returns
        return answer
        
    def add_references(self, question, answer, chunks, verbose=False):
        """
        Adds references to an answer.
        """
        # builds the prompt
        system_message = {"role": "system", "content": ANSWERING_PROMPT}
        context_messages = [{"role": "assistant", "content": str(chunk)} for chunk in chunks]
        question_message = {"role": "user", "content": question}
        answer_message = {"role": "assistant", "content": answer}
        reference_message = {"role": "user", "content": REFERENCE_PROMPT}
        messages = [system_message] + context_messages + [question_message, answer_message, reference_message]
        # runs the query
        references = self.query(messages, verbose=verbose)
        # remove any irrelevant line
        references = keep_references_only(references)
        # updates the answer
        answer = f"{answer}\n\nReferences:\n{references}"
        return answer

    def extract_question(self, previous_messages, verbose=False):
        """
        Extracts the latest question given a list of messages.
        Message are expectted to be dictionnaries with a 'role' ('user' or 'assistant') and 'content' field.
        the question returned will be a string.
        """
        # builds the prompt
        system_message = {"role": "system", "content": QUESTION_EXTRACTION_PROMPT_SYSTEM}
        context_messages = [{'role':'assistant', 'content': f"{m['role']}: {m['content']}"} for m in previous_messages]
        user_message = {"role": "user", "content": QUESTION_EXTRACTION_PROMPT_USER}
        messages = [system_message] + context_messages + [user_message]
        # keep as many context messages as we can
        while self.token_counter_messages(messages) + self.upper_question_size > self.context_size:
            if len(messages) > 3:
                # reduce the context size by popping the oldest context message
                # ensuring there is at least one context message
                messages.pop(1)
            else:
                # no more space to reduce context size
                raise ValueError("You query is too long for the model's context size.")
        # shortcut if there is only one message
        if len(messages) == 3:
            return previous_messages[-1]['content']
        else:
            return self.query(messages, verbose=verbose)