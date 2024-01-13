import re
import torch
from abc import ABC, abstractmethod
from typing import Union, List, Dict
from ...database.document_loader import Chunk
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    Abstraction other large language models
    by default, it is built upon hugginface Transformers library
    """
    def __init__(self, pretrained_model_name_or_path:str, device='cuda'):
        self.pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.pretrained_model_name_or_path, trust_remote_code=False, low_cpu_mem_usage=True, torch_dtype='auto').to(device)
        self.context_size = self.model.config.max_position_embeddings
        self.device = device

    def tokenize(self, input_data: Union[str, List[Dict[str, str]]]) -> torch.Tensor:
        """
        Tokenizes a single string or a conversation (list of dictionaries with 'role' and 'content' fields) using the model's tokenizer.

        Args:
            input_data (Union[str, List[Dict[str, str]]]): Input data to tokenize.

        Returns:
            torch.Tensor: Tensor of token IDs.
        """
        # Check if input is a string or a list of dicts and process accordingly
        if isinstance(input_data, str):
            # Process a single string input
            return self.tokenizer.encode(input_data, return_tensors='pt')
        elif isinstance(input_data, list) and all(isinstance(i, dict) for i in input_data):
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
        return len(input_tokens)

    def query(self, input_data: Union[str, List[Dict[str, str]]], temperature=0.0, verbose=False) -> str:
        """
        Query the model and get a response.

        Args:
            input_data (Union[str, List[Dict[str, str]]]): The input text or conversation history to generate a response for.
            verbose (bool): If True, print additional information.

        Returns:
            str: The generated response from the model.
        """
        # tokenize input
        input_tokens = self.tokenize(input_data).to(self.device)

        # Generate a response from the model
        with torch.no_grad():
            output_tokens = self.model.generate(input_tokens, max_length=self.context_size, temperature=temperature)[0]

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
        Message are expectted to be dictionnaries with a 'role' ('user' or 'assistant') and 'content' field.
        the question returned will be a string.
        """
        pass

from .vicuna import Vicuna
from .llama2 import Llama2