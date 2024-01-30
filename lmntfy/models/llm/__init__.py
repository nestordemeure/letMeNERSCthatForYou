import os
from enum import Enum, auto
from abc import ABC, abstractmethod
from copy import copy
from typing import List, Dict
from ...database.document_loader import Chunk
from transformers import PreTrainedTokenizer
import outlines
from outlines.models.transformers import Transformer
from outlines.generate import SequenceGenerator

# NOTE: this is needed as the initialization of outlines's caching system will try to write to $HOME/.chache/outlines
os.environ['OUTLINES_CACHE_DIR'] = os.environ.get('TMPDIR')
# disables caching
outlines.disable_cache()

#----------------------------------------------------------------------------------------
# MODEL ABSTRACTION

class LanguageModel(ABC):
    """
    Abstraction over large language models
    Built on top of the [Outlines](https://github.com/outlines-dev/outlines) library

    Message format:
        we expect messages to have a "role" ("system", "user", or "assistant") as well as a "content" field
        all "system" messages will be concatenated and put at the beginning of the conversation
        if the conversation is too long to fit the answer, messages with a "relevancy" field will be dropped (starting with lowest relevancy) until it fits
    """
    def __init__(self, pretrained_model_name_or_path:str, device='cuda'):
        self.pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        self.model: Transformer = outlines.models.transformers(self.pretrained_model_name_or_path, device=device)
        self.tokenizer: PreTrainedTokenizer = self.model.tokenizer.tokenizer # get the Transformer Tokenizer for chattemplating purposes
        self.context_size = self.model.model.config.max_position_embeddings
        # generators
        self.base_generator = outlines.generate.text(self.model)

    def count_tokens(self, text:str) -> int:
        """
        Counts the number of tokens in a given string.
        """
        tokens = self.tokenizer.encode(text, return_tensors='pt')
        token_number = tokens.size(-1)
        return token_number

    def _merge_system_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Given a conversation, merge all system messages into the first message.
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

    def _drop_lowest_priority_message(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Finds the message with the lowest relevancy score and drops it from the conversation.
        If no message can be cut, returns None.
        """
        # Find the index of the least relevant message, if any
        least_relevant_index = None
        least_relevancy = float('inf')
        for i, message in enumerate(messages):
            if ('relevancy' in message) and (message['relevancy'] < least_relevancy):
                least_relevancy = message['relevancy']
                least_relevant_index = i

        # returns
        if least_relevant_index is None:
            # no message can be cut, return None
            return None
        else:
            # pop lowest relevancy message
            messages.pop(least_relevant_index)
            return messages

    def apply_chat_template(self, messages: List[Dict[str, str]], nb_tokens_max:int=None) -> str:
        """
        Takes a list of messages and applies the model's chat template.

        NOTE:
        - drops optional messages until the result fits in the given size
        - merge all systems messages into the first message
        """
        # fails hard if the tokeniser does not have a chat_template
        if self.tokenizer.chat_template is None:
            raise RuntimeError(f"Your tokeniser ({type(self.tokenizer)}) of choice does not have a chat_template. See [this repository](https://github.com/chujiezheng/chat_templates/tree/main) for common options.")
        
        # merge system messages (in case there is more than one)
        merged_messages = self._merge_system_messages(messages)
        
        # turns the conversation into a single string
        output_string = self.tokenizer.apply_chat_template(merged_messages, add_generation_prompt=True, tokenize=False)

        # drop optional messages if needed
        if (nb_tokens_max is not None) and (self.count_tokens(output_string) > nb_tokens_max):
            shorten_messages = self._drop_lowest_priority_message(messages)
            if shorten_messages is not None:
                # if we could drop a message
                return self.apply_chat_template(shorten_messages, nb_tokens_max)
            
        return output_string

    def generate(self, text:str, verbose:bool=False, generator:SequenceGenerator=None) -> str:
        """
        Query the model and get a response.

        Args:
            input_data (str): the text prompt
            verbose (bool): should we print debug information? (defaults to False)
            generator (SequenceGenerator): outline generator and the counstraints that go with it

        Returns:
            str: The generated response from the model.
        """
        # picks a generator, defaults to basic text completion
        if generator is None:
            generator = self.base_generator

        # runs the generator
        output = generator(text)

        # debugging information
        if verbose:
            print(text)
            print(output)

        return output

    @abstractmethod
    def extract_question(self, previous_messages:List[Dict], verbose=False) -> str:
        """
        Extracts the user's last question.
        NOTE: this will always return even if the user is just thanking us and done with questions.
        """
        pass

    @abstractmethod
    def chat(self, messages:List[Dict[str, str]], chunks:List[Chunk], verbose=False) -> str:
        """
        Chat with the model given the previous messages
        and relevant chnuks of the documentation to enrich the chat.
        """
        pass

from .vicuna import Vicuna
#from .llama2 import Llama2