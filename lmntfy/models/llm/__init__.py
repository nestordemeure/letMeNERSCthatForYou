from abc import ABC
from copy import copy
from typing import List, Dict
from ...database.document_loader import Chunk
import asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer
from .utilities import StopWordCriteria, validate_references, format_reference_list

#----------------------------------------------------------------------------------------
# PROMPTS

# Basic chat prompt
CHAT_PROMPT_SYSTEM = """\
You are a member of the NERSC supercomputing center's support staff answering a user's questions.
Use an unbiased and journalistic tone. \
Only cite the most relevant results that answer the user's questions accurately. \
Try and be careful not to go off-topics.
"""

# prompt to answer a question
# NOTE: 
# * we do a single shot prompt (with an example answer) to ensure proper formating of the answer at the price of a few tokens
# * note that the end of the prompt is ready to accomodate some chunks of information
# NOTE: we use "concise and informative" instead of "comprehensive and informative" in our previous iteration of the prompt
ANSWERING_PROMPT="""\
You are a member of the NERSC supercomputing center's support staff.
Generate a concise and informative answer for a given question solely based on the provided information (URL and Extract), some of which might be irrelevant (in which case you can simply ignore them).
You must only use information from the provided search results. \
Use an unbiased and journalistic tone. \
Combine search results together into a coherent answer. \
Only cite the most relevant results that answer the question accurately. \
Try and be careful not to go off-topics. \
After providing the answer, list the URLs of the information sources you used in a section called `References:`, sorted from most to least relevant. Include ONLY the URLs that are directly relevant to the answer.

### Example Answer Format:

To optimize your code for CPU usage at NERSC, it's crucial to focus on vectorization and parallelization. Vectorization allows your code to process multiple data points with a single instruction, effectively reducing the time your code takes to run through large datasets. Parallelization, on the other hand, involves dividing your code into multiple tasks that can be processed simultaneously, maximizing the use of available CPU resources. Combining these two strategies can lead to significant improvements in your code's performance on NERSC systems.

References:
 * <https://docs.nersc.gov/performance/vectorization>
 * <https://docs.nersc.gov/performance/parallelism>

### Information Sources:
"""

#----------------------------------------------------------------------------------------
# MODEL ABSTRACTION

# Ensure that not more than one transformer model is currently running on the GPU
transformer_gpu_lock = asyncio.Lock()

class LanguageModel(ABC):
    """
    Abstraction over large language models
    Built on top of the [Outlines](https://github.com/outlines-dev/outlines) library

    See [this leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) for a good list of current models that you might want to use.

    Message format:
        we expect messages to have a "role" ("system", "user", or "assistant") as well as a "content" field
        all "system" messages will be concatenated and put at the beginning of the conversation
        if the conversation is too long to fit the answer, messages with a "relevancy" field will be dropped (starting with lowest relevancy) until it fits

        if `use_system_prompt` is False, then the system prompt will be converted into a starting message
    """
    def __init__(self, pretrained_model_name_or_path:str, model_kwargs:dict=dict(), use_system_prompt=True, device='cuda'):
        self.pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.pretrained_model_name_or_path, device_map=device, **model_kwargs)
        self.context_size = self.model.config.max_position_embeddings
        self.upper_answer_size = None # needs to be filled per tokenizer
        self.upper_question_size = None # needs to be filled per tokenizer
        self.use_system_prompt = use_system_prompt
        # prompts
        self.CHAT_PROMPT_SYSTEM = CHAT_PROMPT_SYSTEM
        self.ANSWERING_PROMPT = ANSWERING_PROMPT

    def count_tokens(self, text:str) -> int:
        """
        Counts the number of tokens in a given string.
        """
        tokens = self.tokenizer.encode(text, return_tensors='pt')
        token_number = tokens.size(-1)
        return token_number

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
        # fails hard if the tokeniser does not have a chat_template
        if self.tokenizer.chat_template is None:
            raise RuntimeError(f"Your tokeniser ({type(self.tokenizer)}) of choice does not have a chat_template. See [this repository](https://github.com/chujiezheng/chat_templates/tree/main) for common options.")
        
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

    async def generate(self, prompt:str, stopwords:List[str]=[], strip_stopword:bool=True, verbose:bool=False) -> str:
        """
        Query the model and get a response.
        NOTE: 
        * this function is written to deal with a single piece of text, not a batch
        * this function is not async

        Args:
            prompt (str): the text prompt
            stopwords (List[str]): the words on which to stop the generation, if any
            strip_stopword (bool): should we strip the stopword from our output (default to True)
            verbose (bool): should we print debug information? (defaults to False)

        Returns:
            str: The generated response from the model.
        """
        # used to stop on the stop words
        stopping_criteria = StopWordCriteria(tokenizer=self.tokenizer, prompts=[prompt], stop_words=stopwords)

        # tokenize the input text
        inputs_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # runs the LLM, producing tokens for output=input+answer+stopword+?
        #output_tokens = self.model.generate(inputs_tokens, 
        #                                    max_length=self.context_size, 
        #                                    pad_token_id=self.tokenizer.eos_token_id,
        #                                    stopping_criteria=[stopping_criteria])
        # NOTE: we ensure that only one request is currently running on the GPU
        #       meanwhile, other CPU tasks can be done
        #       -> we could cut the code and have this engine be actualy synchronous (but this would be bad)
        async with transformer_gpu_lock:
            output_tokens = await asyncio.to_thread(self.model.generate, 
                                                    inputs_tokens, 
                                                    max_length=self.context_size, 
                                                    pad_token_id=self.tokenizer.eos_token_id,
                                                    stopping_criteria=[stopping_criteria])

        # extract answer text from output tokens, cutting prompt and stop words
        answer = stopping_criteria.extract_answers(output_tokens, strip_stopword=strip_stopword)[0]

        # debugging information
        if verbose: print(f"{prompt}\n{answer}")
        return answer

    async def extract_question(self, previous_messages:List[Dict], verbose=False) -> str:
        """
        Tries to extract the last question.
        """
        # shortcut for single message conversations
        if len(previous_messages) == 1:
            return previous_messages[-1]['content']
        # builds the messages
        system_message = {"role": "system", "content": self.CHAT_PROMPT_SYSTEM}
        formatted_discussion = [{**message, 'relevancy': i} for (i,message) in enumerate(previous_messages)]
        messages = [system_message] + formatted_discussion
        # builds the base prompt
        prompt = self.apply_chat_template(messages, nb_tokens_max=self.context_size-self.upper_question_size)
        # prime the model to extract the question
        prompt_question_extraction = prompt + 'If I understand you clearly, your question is: "'
        question = await self.generate(prompt_question_extraction, stopwords=['"'], verbose=verbose)
        return question

    async def chat(self, discussion:List[Dict[str, str]], chunks:List[Chunk], verbose=False) -> str:
        """
        Chat with the model given the previous messages
        and relevant chunks of the documentation to enrich the chat.
        """
        # builds the messages
        nb_messages_minimum = 3 # keep at least that many messages (when possible)
        nb_relevant_messages = len(chunks) + max(0, len(discussion)-nb_messages_minimum)
        system_message = {"role": "system", "content": self.ANSWERING_PROMPT}
        # formats the chunks
        chunks_messages = [{"role": "system", "content": f"\n{chunk.to_markdown()}", "relevancy": (nb_relevant_messages-i)} for (i,chunk) in enumerate(chunks)]
        # formats the discussion
        if len(discussion) <= nb_messages_minimum:
            discussion_messages = discussion
        else:
            # extracts the last 3 messages, they will stay untouched
            discussion_end = discussion[-3:]
            # extract the first messages and add a relevancy to them
            discussion_start = discussion[:-3]
            discussion_start = [{**message, 'relevancy': i} for (i,message) in enumerate(discussion_start)]
            # assemble the discussion
            discussion_messages = discussion_start + discussion_end
        # assemble the messages
        messages = [system_message] + chunks_messages + discussion_messages

        # turns the messages into a prompt
        prompt = self.apply_chat_template(messages, nb_tokens_max=self.context_size-self.upper_answer_size)

        # generates an answer, stopping at the reference section
        reference_section_titles = ["References:", "Reference(s):", "Sources:", "Ressources:", "Source URL:", "Source URLs:"]
        answer_body = await self.generate(prompt, stopwords=reference_section_titles, verbose=verbose)
        # generate a reference section to go with the answer
        reference_section = await self.add_references(prompt + answer_body, chunks, verbose=verbose)
        # assemble the answer
        return answer_body + '\n\n' + reference_section

    async def add_references(self, original_prompt: str, chunks: List[Chunk], verbose: bool = False) -> str:
        """
        Generates a reference section for a given prompt using specified documentation chunks.

        Args:
            original_prompt (str): The text generated so far, including the initial prompt.
            chunks (List[Chunk]): A list of documentation data chunks to be referenced.
            verbose (bool, optional): If True, enables verbose output. Defaults to False.

        Returns:
            str: The prompt text appended with a formatted reference section containing URLs.
        """
        # prompt the model to start writing a properly formated reference section
        prompt = original_prompt + "\n\nReferences:\n * <"

        # gets at most one url per chunk
        urls = []
        for _ in chunks:
            # Generate a reference, stops if it is done or ready to start a new reference
            generated_reference = await self.generate(prompt, stopwords=['<'], strip_stopword=False, verbose=verbose)
            # Update the prompt for the next generation
            prompt += generated_reference
            # Extracts the url
            url = generated_reference.split('>', 1)[0]
            urls.append(url)
            # Break the loop if the model is not getting ready to start on a new reference
            if not generated_reference.endswith('<'):
                break

        # Validate and filter URLs.
        valid_urls = validate_references(urls, chunks, original_prompt)
        # Builds a reference section with only the validated URLs.
        reference_section = "References:\n" + format_reference_list(valid_urls)
        return reference_section

# Transformers
from .llama2 import Llama2 #hallucinate often
from .vicuna import Vicuna #good but I have had some cut-offs problems
from .mistral import Mistral #good at answering, not at picking references
from .zephyr import Zephyr #good but can miss some information from the doc provided
from .codellama import CodeLlama #good answers but does not care much for the provided doc
from .mixtral import Mixtral #too heavy for local serving
from .gemma import Gemma # tends to answer not quite the question asked TODO to be reevaluated after transformer update
from .openchat import OpenChat # answers somewhat in league with mistral
from .qwen import Qwen # really nice (feels competitive with mistral)
from .snorkel import Snorkel # good answers but high hallucinations
from .starling import Starling # a bit verbose but very good
from .starling import StarlingCode # not as good as base (understandable as this one is for code writing only)
# vLLM
from .utilities.vllm_backend import MistralVllm
# default model
Default = Mistral
