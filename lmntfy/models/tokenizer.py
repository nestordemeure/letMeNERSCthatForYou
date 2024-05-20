from abc import ABC
from copy import copy
from typing import List, Dict
from transformers import AutoTokenizer, LlamaTokenizerFast, Qwen2TokenizerFast, PreTrainedTokenizerFast

#--------------------------------------------------------------------------------------------------
# GENERALIST

class Tokenizer(ABC):
    """
    Encapsulates a model's tokenizer and the operations we need from it:
    * counting tokens
    """
    def __init__(self, pretrained_model_name_or_path:str, context_size:int=None):
        self.pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        self.name = self.tokenizer.__class__.__name__
        self.context_size = context_size

    def count_tokens(self, text:str) -> int:
        """
        Counts the number of tokens in a given string.
        """
        tokens = self.tokenizer.encode(text, return_tensors='pt')
        token_number = tokens.size(-1)
        return token_number

#--------------------------------------------------------------------------------------------------
# CHAT

class ChatTokenizer(Tokenizer):
    """
    Encapsulates a chat-model's tokenizer and the operations we need from it:
    * counting tokens
    * applying a chat template

    Message format:
        we expect messages to have a "role" ("system", "user", or "assistant") as well as a "content" field
        all "system" messages will be concatenated and put at the beginning of the conversation
        if the conversation is too long to fit the answer, messages with a "relevancy" field will be dropped (starting with lowest relevancy) until it fits

        if `use_system_prompt` is False, then the system prompt will be converted into a starting message
    """
    def __init__(self, pretrained_model_name_or_path:str, context_size:int=None, chat_template:str=None, use_system_prompt:bool=True):
        super().__init__(pretrained_model_name_or_path, context_size)
        # loads chat specific, tokenizer-dependant parameters
        self.use_system_prompt = use_system_prompt
        self._set_upper_sizes()
        self._set_chat_template(chat_template)

    def _set_upper_sizes(self):
        """
        Sets self.upper_answer_size and self.upper_question_size as a function of the tokenizer type.
        """
        if (self.tokenizer is None):
            raise RuntimeError("You need to initialize the tokenizer before you can decide on appropriate upper sizes for it.")
        elif isinstance(self.tokenizer, LlamaTokenizerFast):
            # most llama and mistral based models
            self.upper_answer_size = 450
            self.upper_question_size = 200
        elif isinstance(self.tokenizer, PreTrainedTokenizerFast):
            # llama3
            self.upper_answer_size = 400
            self.upper_question_size = 180
        elif isinstance(self.tokenizer, Qwen2TokenizerFast):
            # qwen models
            self.upper_answer_size = 400
            self.upper_question_size = 180
        else:
            raise RuntimeError(f"Please use the token_counter.py script to find out proper upper size for your tokenizer (of type {type(self.tokenizer)}).")

    def _set_chat_template(self, chat_template=None):
        """
        Insures we have a chat template.
        """
        if chat_template is not None:
            # sets the chat template
            self.tokenizer.chat_template = chat_template
        elif self.tokenizer.chat_template is None:
            # fails hard if the tokeniser does not have a chat_template
            raise RuntimeError(f"Your tokeniser ({type(self.tokenizer)}) of choice does not have a chat_template. See [this repository](https://github.com/chujiezheng/chat_templates/tree/main) for common options.")

    def _clean_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Given a conversation:
        * merge all system messages into the first message
        * ensure they alternate properly between user and assistant (can be caused by relevancy problems)
          by dropping eroneous messages silently
        """
        # gets a system message
        if (len(messages) < 1) or (messages[0]['role'] != "system"):
            # NOTE: no starting system message, we assume that this moel does not use system messages
            return messages
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

        # ensure alternance of user-assistant messages in non-system messages
        result = [system_message]
        next_is_user = True
        for message in nonsystem_messages:
            is_user = (message['role'] == 'user')
            if (next_is_user and is_user) or ((not next_is_user) and (not is_user)):
                result.append(message)
                next_is_user = not next_is_user
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
        # merge system messages (in case there is more than one)
        merged_messages = self._clean_messages(messages)
        
        # in the absence of a system prompt, converts the system prompt
        if not self.use_system_prompt:
            # extracts the system message
            system_message = merged_messages[0]
            chat_messages = merged_messages[1:]
            # turn the system message into a user message followed by ok as system messages are not allowed by the model
            prompt_message = {'role':'user', 'content':system_message['content']}
            okay_message = {'role':'assistant', 'content':"Understood! From now on I am a member of the NERSC supercomputing center's support staff discussing with a user (you)."}
            merged_messages = [prompt_message, okay_message] + chat_messages

        # turns the conversation into a single string
        try:
            output_string = self.tokenizer.apply_chat_template(merged_messages, add_generation_prompt=True, tokenize=False)
        except Exception as e:
            raise RuntimeError(f"Failed to apply chat templates with error '{e}'. Roles are: {[m['role'] for m in merged_messages]}") from e

        # drop optional messages if needed
        if (nb_tokens_max is not None) and (self.count_tokens(output_string) > nb_tokens_max):
            shorten_messages = self._drop_lowest_priority_message(messages)
            if shorten_messages is not None:
                # if we could drop a message
                return self.apply_chat_template(shorten_messages, nb_tokens_max)
            
        return output_string