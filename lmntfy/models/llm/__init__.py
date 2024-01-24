import re
import torch
from abc import ABC, abstractmethod
from copy import copy
from typing import Union, List, Dict
from ...database.document_loader import Chunk
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

def keep_references_only(input_str: str) -> str:
    """
    Extracts and formats URLs from a given string into a bullet list. It identifies all URLs, removes any trailing '>'
    often found in '<url>' patterns, eliminates duplicates while preserving order, and then formats them into a 
    bullet list.

    Args:
        input_str (str): The string from which URLs are to be extracted.

    Returns:
        str: A bullet list of unique URLs found in the input string.
    """
    # extract all urls
    url_pattern = re.compile(r'https?://(?:[a-zA-Z]|[0-9]|[-.#/]|[$@&+]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    urls = re.findall(url_pattern, input_str)
    # remove trailing `>` (from <url> types of pattern)
    urls = [url[:-1] if url.endswith('>') else url for url in urls]
    # remove duplicates while preserving order
    urls = list(dict.fromkeys(urls))
    # convert list into a bullet list of URLs
    bullet_list = "\n".join(f"* <{url}>" for url in urls)
    return bullet_list

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
        # update the generation config to a Greedy search
        # (equivalent to temperature=0 but faster)
        # NOTE: we set temperature and top_p to there default values as a way to unset them
        self.model.generation_config.update(do_sample=False, temperature=1.0, top_p=1.0)

    def _merge_systems(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        given a conversation, merge all system messages into the first message.
        """
        # gets a system message
        if (len(messages) < 1) or (messages[0]['role'] != "system"):
            raise RuntimeError("Your messages do not start with a system prompt!")
        else:
            system_message = copy(messages[0])
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
            print("Input:", input_data)
            print("Response:", answer_string)
        return answer_string

    @abstractmethod
    def get_answer(self, question:str, chunks:List[Chunk], verbose=False) -> str:
        """
        Abstract method to get an answer given a question and some chunks passed for context.
        """
        pass

    @abstractmethod
    def extract_question(self, messages:List[Dict], verbose=False) -> str:
        """
        Abstract method to extract the latest question given a list of messages.
        Message are expected to be dictionnaries with a 'role' ('user' or 'assistant') and 'content' field.
        the question returned will be a string.
        """
        pass

from .vicuna import Vicuna
from .llama2 import Llama2
