import re
import tiktoken
import openai
from . import LanguageModel
from .. import retry

#----------------------------------------------------------------------------------------
# PROMPTS

# You are a helpful member of the NERSC support staff.
ANSWERING_PROMPT="Generate a comprehensive and informative answer (but no more than 80 words) for a given question solely based on the provided web Search Results (URL and Extract). You must only use information from the provided search results. Use an unbiased and journalistic tone. Combine search results together into a coherent answer. Cite search results using [${number}] notation. Only cite the most relevant results that answer the question accurately."

def format_chunk(chunk, index):
    """takes  chunk and format it to include its index and source in the message"""
    return f"URL {index}: {chunk['source']}\n\n{chunk['content']}"

def add_references(answer, chunks, verbose=False):
    # TODO we need to shift references inside the text
    # TODO deal with several reference together in a coma separated list: [a, b, c]
    if verbose:
        references = [(i+1) for i in range(len(chunks))]
    else:
        # Regular expression pattern to match references in the form [${number}] or [number]
        pattern = re.compile(r'\[\$?\{?(\d+)\}?\]')
        # Find all matches and convert them to reference index
        references = list({int(match) for match in pattern.findall(answer)})
        references.sort()
    # add references at the end of the answer
    if len(references) > 0:
        answer += '\n'
        for reference in references:
            source = chunks[reference-1]['source']
            answer += f"\n[{reference}]: {source}"
    return answer

#----------------------------------------------------------------------------------------
# MODEL

class GPT35(LanguageModel):
    def __init__(self, 
                 model_name='gpt-3.5-turbo', 
                 context_size=4096):
        super().__init__(model_name, context_size)
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.model_tokens_per_message = 4
        self.model_tokens_per_name = -1
        self.upper_answer_size = 250

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
        response = openai.ChatCompletion.create(model=self.model_name, messages=messages)
        if verbose: print(messages + [response.choices[0].message])
        return response.choices[0].message['content']

    def get_answer(self, question, chunks, verbose=False):
        """
        Method to get an answer given a question and some chunks passed for context.
        """
        system_message = {"role": "system", "content": ANSWERING_PROMPT}
        question_message = {"role": "user", "content": question}
        context_messages = [{"role": "assistant", "content": format_chunk(chunk,i+1)} for (i,chunk) in enumerate(chunks)]
        # builds the answer with as many context messages as we can
        messages = [system_message] + context_messages + [question_message]
        while self.token_counter_messages(messages) + self.upper_answer_size > self.context_size:
            if len(context_messages) > 0:
                # reduce the context size
                context_messages.pop()
            else:
                # no more space to reduce context size
                raise ValueError("You query is too long for the model's context size.")
        # runs the query
        answer = self.query(messages, verbose=verbose)
        # adds sources at the end of the query
        return add_references(answer, chunks, verbose=verbose)
        