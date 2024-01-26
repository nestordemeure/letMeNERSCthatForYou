import torch
from enum import Enum, auto
from abc import ABC, abstractmethod
from copy import copy
from typing import Union, List, Dict
from ...database.document_loader import Chunk
from transformers import AutoTokenizer, AutoModelForCausalLM

#----------------------------------------------------------------------------------------
# TRIAGE ANSWER TYPE

class AnswerType(Enum):
    OUT_OF_SCOPE = auto()
    QUESTION = auto()
    SMALL_TALK = auto()

class Answer:
    """output of the triage operation"""
    def __init__(self, answer_type:AnswerType, content:str = None, raw:str = None):
        self.answer_type = answer_type
        self.content = content
        self.raw = raw

    @classmethod
    def out_of_scope(cls, raw:str=None):
        return cls(AnswerType.OUT_OF_SCOPE, raw=raw)

    @classmethod
    def question(cls, question:str, raw:str=None):
        return cls(AnswerType.QUESTION, content=question, raw=raw)

    @classmethod
    def smallTalk(cls, answer:str, raw:str=None):
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

#----------------------------------------------------------------------------------------
# MODEL ABSTRACTION

class LanguageModel(ABC):
    """
    Abstraction over large language models
    by default, it is built upon hugginface Transformers library

    format:
        we expect messages to have a "role" ("system", "user", or "assistant") as well as a "content" field
        all "system" messages will be concatenated and put at the beginning of the conversation
        if the conversation is too long to fit the answer, messages with a "relevancy" field will be dropped (starting with lowest relevancy) until it fits
    """
    def __init__(self, pretrained_model_name_or_path:str, device='cuda'):
        self.pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.pretrained_model_name_or_path, 
                                                          trust_remote_code=False, low_cpu_mem_usage=True, torch_dtype='auto').to(device)
        self.context_size = self.model.config.max_position_embeddings
        self.device = device
        # Pick our prefered Generation Strategy
        # see: https://huggingface.co/docs/transformers/generation_strategies
        self.model.generation_config.update(do_sample=False, # greedy generation (<=> temp=0)
                                            temperature=1.0, top_p=1.0) # unset by setting to default

    def _merge_systems(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        given a conversation, merge all system messages into the first message.
        """
        # gets a system message
        if (len(messages) < 1) or (messages[0]['role'] != "system"):
            raise RuntimeError("Your messages do not start with a system prompt!")
        else:
            system_message = copy(messages[0])
            messages = messages[1:]
        # accumulate system messages
        nonsystem_messages = []
        for message in messages:
            if (message['role'] == "system"):
                # add content to the system message
                system_message['content'] += message['content']
            else:
                # add message to result
                nonsystem_messages.append(message)
        # builds result starting with a system message
        result = [system_message] + nonsystem_messages
        return result

    def tokenize(self, input_data: Union[str, List[Dict[str, str]]]) -> torch.Tensor:
        """
        Tokenizes a single string or a conversation (list of dictionaries with 'role' and 'content' fields) using the model's tokenizer.

        Args:
            input_data (Union[str, List[Dict[str, str]]]): Input data to tokenize. 

        Returns:
            torch.Tensor: Tensor of token IDs.
        
        NOTE: if the input is a conversation with several system messages, they will be concatenated into one before tokenisation
        """
        # Check if input is a string or a list of dicts and process accordingly
        if isinstance(input_data, str):
            # Process a single string input
            return self.tokenizer.encode(input_data, return_tensors='pt')
        elif isinstance(input_data, list) and all(isinstance(i, dict) for i in input_data):
            # fails hard if the tokeniser does not have a chat_template
            if self.tokenizer.chat_template is None:
                raise RuntimeError(f"Your tokeniser ({type(self.tokenizer)}) of choice does not have a chat_template. See [this repository](https://github.com/chujiezheng/chat_templates/tree/main) for common options.")
            # merge system messages (in case there is more than one)
            input_data = self._merge_systems(input_data)
            # Process a conversation represented as a list of dictionaries
            return self.tokenizer.apply_chat_template(conversation=input_data, tokenize=True, return_tensors='pt')
        else:
            raise ValueError("Input data must be either a string or a list of dictionaries.")

    def token_counter(self, input_data: Union[str, List[Dict[str, str]]]) -> int:
        """
        Counts the number of tokens in a string or a conversation (list of dictionaries).

        Args:
            input_data (Union[str, List[Dict[str, str]]]): Input data to be tokenized.

        Returns:
            int: Count of tokens in the input.
        """
        input_tokens = self.tokenize(input_data)
        return input_tokens.size(-1)

    def _drop_irrelevant_messages(self, messages: List[Dict[str, str]], expected_answer_size: int) -> List[Dict[str, str]]:
        """
        Takes a conversation and insures it, and the answer, will fit in the context length
        Does so by droppping as many messages with a "relevancy" field as necessary (from least relevant to most relevant)
        """
        # Calculate the size of the current conversation
        conversation_size = self.token_counter(messages)

        # Check if the conversation size is within the valid range
        if (conversation_size + expected_answer_size) <= self.context_size:
            return messages

        # Find the index of the least relevant message, if any
        least_relevant_index = None
        least_relevancy = float('inf')
        for i, message in enumerate(messages):
            if 'relevancy' in message:
                if message['relevancy'] < least_relevancy:
                    least_relevancy = message['relevancy']
                    least_relevant_index = i

        # If there are no messages with a "relevancy" field or the least relevant message is found, remove it
        if least_relevant_index is not None:
            messages.pop(least_relevant_index)
        else:
            # If there are no messages with "relevancy" left, return the conversation as is
            return messages

        # Recursively call trim_conversation if needed
        return self._drop_irrelevant_messages(messages, expected_answer_size)

    def query(self, input_data: Union[str, List[Dict[str, str]]], expected_answer_size=None, verbose=False) -> str:
        """
        Query the model and get a response.

        Args:
            input_data (Union[str, List[Dict[str, str]]]): The input text or conversation history to generate a response for.
            expected_answer_size (int): how long (in number of tokens) do we expect the answer to be? (default to None)
            verbose (bool): If True, print additional information (defaults to False).

        Returns:
            str: The generated response from the model.
        """
        # drop irrelevant messages to leave space for the answer
        if (expected_answer_size is not None) and isinstance(input_data, list):
            input_data = self._drop_irrelevant_messages(input_data, expected_answer_size)

        # tokenize input
        input_tokens = self.tokenize(input_data).to(self.device)

        # Generate a response from the model
        with torch.no_grad():
            max_new_tokens = self.context_size - input_tokens.size(-1)
            output_tokens = self.model.generate(input_tokens, max_new_tokens=max_new_tokens)[0]

        # keep only the answer and not the full conversation
        answer_tokens = output_tokens[input_tokens.size(-1):]
        # Decode the answer tokens to a string
        answer_string = self.tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()

        # returns
        if verbose:
            output_string = self.tokenizer.decode(output_tokens, skip_special_tokens=True).strip()
            print(f"Conversation:\n\n{output_string}")
        return answer_string

    @abstractmethod
    def triage(self, messages:List[Dict], verbose=False) -> Answer:
        """
        Decides whether the message is:
        * out of scope,
        * a normal discussion (ie: "thank you!") that does not require a documentation call,
        * something that requires a documentation call
        """
        pass

    @abstractmethod
    def get_answer(self, question:str, chunks:List[Chunk], verbose=False) -> str:
        """
        Abstract method to get an answer given a question and some chunks passed for context.
        """
        pass

from .vicuna import Vicuna
#from .llama2 import Llama2