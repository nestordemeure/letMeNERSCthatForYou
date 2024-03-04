import os
import re
import string
from abc import ABC
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

# regex that produces a bullet list of, at most, 10 urls
# NOTE: we purposefully allow non-NERSC url
#       if the system produces garbage we want easily deleted random urls rather than previous valid NERSC ones
REFLIST_REGEX = r"( \* \<([^\>]*?)\>\n){1,10}\n"

#----------------------------------------------------------------------------------------
# MODEL ABSTRACTION

def stem_url(url: str) -> str:
    """Takes a URL and cuts it at the latest '#' if possible.
    
    Args:
        url (str): The URL to be processed.
        
    Returns:
        str: The stemmed URL without the fragment part after the latest '#'.
    """
    # Find the position of the last occurrence of '#'
    hash_position = url.rfind('#')
    # If '#' is found, return the URL up to (but not including) the '#'
    if hash_position != -1:
        return url[:hash_position]
    # If '#' is not found, return the original URL
    return url

def validate_references(references:str, chunks:List[Chunk], prompt:str, stem_before_validation=False) -> str:
    """
    Takes:
    - references: a string with references for the current answer
    - chunks: a list of chnuks used to build the answer
    - prompt: the prompt which contains the conversation so far
    - stem_before_validation (bool): should we ignore the # section at the end of the url?

    A reference is only valid if its root (minus any last minute '#' paragraph):
    - is a chunk's url,
    - or appears inside a chunk,
    - or was referenced in a previous message,
    - AND be in a NERSC domain.

    Returns "https://docs.nersc.gov/" if no valid reference is found.
    """
    # the stemmer function
    stemmer = stem_url if stem_before_validation else (lambda url: url)
    # all urls in prompts
    chunk_urls = {stemmer(chunk.url) for chunk in chunks}
    # all urls in the conversation so far
    reference_pattern = r"\* \<([^\>]*?)\>"
    prompt_urls = {stemmer(url) for url in re.findall(reference_pattern, prompt)}
    # all urls that will be accepted in the output
    valid_urls = chunk_urls | prompt_urls

    # all urls in the references
    references_urls = re.findall(reference_pattern, references)

    # keep only urls referenced or appearing inside a chunk
    urls = set()
    for url in references_urls:
        stemmed_url = stemmer(url)
        if (url.startswith('https://docs.nersc.gov') or url.startswith('https://nersc.gov')) and ((stemmed_url in valid_urls) or any((stemmed_url in chunk.content) for chunk in chunks)):
                urls.add(url)

    if len(urls) == 0:
        # default (useless) references used if no reference is valid
        return " * <https://docs.nersc.gov/>\n * <https://www.nersc.gov/users/getting-help/online-help-desk/>"
    else:
        # builds list of references
        references = [f" * <{url}>" for url in urls]
        # Joins the lines and returns
        return '\n'.join(references)

class LanguageModel(ABC):
    """
    Abstraction over large language models
    Built on top of the [Outlines](https://github.com/outlines-dev/outlines) library

    Message format:
        we expect messages to have a "role" ("system", "user", or "assistant") as well as a "content" field
        all "system" messages will be concatenated and put at the beginning of the conversation
        if the conversation is too long to fit the answer, messages with a "relevancy" field will be dropped (starting with lowest relevancy) until it fits

        if `use_system_prompt` is False, then the system prompt will be converted into a starting message
    """
    def __init__(self, pretrained_model_name_or_path:str, model_kwargs:dict=dict(), use_system_prompt=True, device='cuda'):
        self.pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        self.model: Transformer = outlines.models.transformers(self.pretrained_model_name_or_path, model_kwargs=model_kwargs, device=device)
        self.tokenizer: PreTrainedTokenizer = self.model.tokenizer.tokenizer # get the Transformer Tokenizer for chattemplating purposes
        self.context_size = self.model.model.config.max_position_embeddings
        self.upper_answer_size = None # needs to be filled per tokenizer
        self.upper_question_size = None # needs to be filled per tokenizer
        self.use_system_prompt = use_system_prompt
        # generators
        self.base_generator = outlines.generate.text(self.model)
        self.reflist_generator = outlines.generate.regex(self.model, REFLIST_REGEX)

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

    def extract_question(self, previous_messages:List[Dict], verbose=False) -> str:
        """
        Tries to extract the last question.
        """
        # shortcut for single message conversations
        if len(previous_messages) == 1:
            return previous_messages[-1]['content']
        # builds the messages
        system_message = {"role": "system", "content": CHAT_PROMPT_SYSTEM}
        formatted_discussion = [{**message, 'relevancy': i} for (i,message) in enumerate(previous_messages)]
        messages = [system_message] + formatted_discussion
        # builds the base prompt
        prompt = self.apply_chat_template(messages, nb_tokens_max=self.context_size-self.upper_question_size)
        # prime the model to extract the question
        prompt_question_extraction = prompt + 'If I understand you clearly, your question is: "'
        question = self.base_generator(prompt_question_extraction, stop_at='"')[:-1]
        return question

    def extract_keywords(self, previous_messages:List[Dict], verbose=False) -> str:
        """
        Tries to extract relevant search keywords
        """
        # builds the messages
        system_message = {"role": "system", "content": CHAT_PROMPT_SYSTEM}
        formatted_discussion = [{**message, 'relevancy': i} for (i,message) in enumerate(previous_messages)]
        messages = [system_message] + formatted_discussion
        # builds the base prompt
        prompt = self.apply_chat_template(messages, nb_tokens_max=self.context_size-self.upper_question_size)
        # prime the model to extract the keywords
        prompt_keyword_extraction = prompt + 'Search for the following in the NERSC documentation\'s search bar; it will give you the relevant pages: "'
        keywords = self.base_generator(prompt_keyword_extraction, stop_at='"')[:-1]
        # redo it to get some synonyms
        prompt_keyword_extraction2 = prompt_keyword_extraction + keywords + '"\nDo not hesitate to use synonyms or more general terms in case the keywords you used are not present on the page with your answer. For example: "'
        keywords2 = self.base_generator(prompt_keyword_extraction2, stop_at='"')[:-1]
        # even more
        prompt_keyword_extraction3 = prompt_keyword_extraction2 + keywords2 + '\" or: "'
        keywords3 = self.base_generator(prompt_keyword_extraction3, stop_at='"')[:-1]
        # merge keywords
        keywords = f"{keywords} {keywords2} {keywords3}"
        # remove punctuation
        no_punctuation_translator = str.maketrans('', '', string.punctuation)
        keywords = keywords.translate(no_punctuation_translator)
        return keywords

    def chat(self, discussion:List[Dict[str, str]], chunks:List[Chunk], verbose=False) -> str:
        """
        Chat with the model given the previous messages
        and relevant chunks of the documentation to enrich the chat.
        """
        # builds the messages
        nb_messages_minimum = 3 # keep at least that many messages (when possible)
        nb_relevant_messages = len(chunks) + max(0, len(discussion)-nb_messages_minimum)
        system_message = {"role": "system", "content": ANSWERING_PROMPT}
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
        if verbose: print(f"PROMPT: {prompt}")

        # generates an answer in two part to ensure it follows our prefered format
        # 1. body of the answer
        reference_section_titles = ["References:", "Sources:", "Ressources:", "Source URL:"]
        answer_body = self.base_generator(prompt, stop_at=reference_section_titles)
        # Normalize reference section title
        for title in reference_section_titles:
            answer_body = answer_body.replace(title, "References:")
        # check for the presence of a reference section
        if not "References:" in answer_body: 
            if any(substr in answer_body for substr in ["\n * [", "\n * <", "\n* [", "\n* <"]):
                # there are already references in the answer, exit
                return answer_body
            else:
                # no references found, let's add some
                answer_body += "\n\nReferences:"
        # 2. references, generating following our prefered format
        prompt_extended = prompt + answer_body + '\n'
        answer_references = self.reflist_generator(prompt_extended)
        # keep only valid references
        answer_references = validate_references(answer_references, chunks, prompt)
        # assemble the answer
        answer = answer_body + '\n' + answer_references

        return answer

from .llama2 import Llama2 #hallucinate often
from .vicuna import Vicuna #good but I have had some cut-offs problems
from .mistral import Mistral #good at answering, not at picking references
from .zephyr import Zephyr #good but can miss some information from the doc provided
from .codellama import CodeLlama #good answers but does not care much for the provided doc
from .mixtral import Mixtral #too heavy for local serving
from .gemma import Gemma # tends to answer not quite the question asked
from .openchat import OpenChat # answers somewhat in league with mistral
# the default model
Default = Mistral