from pathlib import Path
from typing import List, Dict
from . import LanguageModel

#----------------------------------------------------------------------------------------
# PROMPTS

# Jinja chat template for Llama2
# found [here](https://github.com/chujiezheng/chat_templates/blob/main/chat_templates/llama-2-chat.jinja)
LLAMA2_CHAT_TEMPLATE = """
{% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = '<<SYS>>\n' + messages[0]['content'].strip() + '\n<</SYS>>\n\n' %}
{% else %}
    {% set loop_messages = messages %}
    {% set system_message = '' %}
{% endif %}
{% for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}
    {% if loop.index0 == 0 %}
        {% set content = system_message + message['content'] %}
    {% else %}
        {% set content = message['content'] %}
    {% endif %}
    {% if message['role'] == 'user' %}
        {{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}
    {% elif message['role'] == 'assistant' %}
        {{ ' '  + content.strip() + ' ' + eos_token }}
    {% endif %}
{% endfor %}
"""

# Basic chat prompt
CHAT_PROMPT_SYSTEM = """
You are a member of the NERSC supercomputing center's support staff answering a user's questions.
Use an unbiased and journalistic tone. \
Only cite the most relevant results that answer the user's questions accurately. \
Try and be careful not to go off-topics.
"""

# prompt to answer a question
# NOTE: 
# * we do a single shot prompt (with an example answer) to ensure proper formating of the answer at the price of a few tokens
# * note that the end of the prompt is ready to accomodate some chunks of information
ANSWERING_PROMPT="""
You are a member of the NERSC supercomputing center's support staff.
Generate a comprehensive and informative answer for a given question solely based on the provided information (URL and Extract).
You must only use information from the provided search results. \
Use an unbiased and journalistic tone. \
Combine search results together into a coherent answer. \
Only cite the most relevant results that answer the question accurately. \
Try and be careful not to go off-topics. \
After providing the answer, list the URLs of the information sources you used in a `References:` section, sorted from most to least relevant. Include ONLY the URLs that are directly relevant to the answer.

### Example Answer Format:

To optimize your code for CPU usage at NERSC, it's crucial to focus on vectorization and parallelization. Vectorization allows your code to process multiple data points with a single instruction, effectively reducing the time your code takes to run through large datasets. Parallelization, on the other hand, involves dividing your code into multiple tasks that can be processed simultaneously, maximizing the use of available CPU resources. Combining these two strategies can lead to significant improvements in your code's performance on NERSC systems.

References:
* <https://docs.nersc.gov/cpu-optimization>
* <https://docs.nersc.gov/parallel-computing>

### Information Sources:
"""

#----------------------------------------------------------------------------------------
# MODEL

class Llama2(LanguageModel):
    def __init__(self, 
                 models_folder: Path,
                 model_name: str='Llama-2-13b-chat-hf',
                 device='cuda'):
        super().__init__(models_folder / model_name, device=device)
        self.tokenizer.chat_template = LLAMA2_CHAT_TEMPLATE
        self.upper_answer_size = 450
        self.upper_question_size = 200

    def extract_question(self, previous_messages:List[Dict], verbose=False) -> str:
        """
        Tries to extract the last question.
        """
        # builds the messages
        system_message = {"role": "system", "content": CHAT_PROMPT_SYSTEM}
        formatted_discussion = [{**message, 'relevancy': i} for (i,message) in enumerate(previous_messages)]
        messages = [system_message] + formatted_discussion
        # builds the base prompt
        prompt = self.apply_chat_template(messages, nb_tokens_max=self.context_size-self.upper_question_size)
        # prime the model to extract the question
        prompt_question_extraction = prompt + 'If I understand you clearly, your question is: "'
        question = self.base_generator(prompt_question_extraction, stop_at='"')[:-1]
        print(f"DEBUGGING: question:'{question}'")
        return question

    def chat(self, discussion:List[Dict[str, str]], chunks:List, verbose=False) -> str:
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

        # generates an answer in two part to ensure it follows our prefered format
        # 1. body of the answer
        answer_body = self.base_generator(prompt, stop_at="References:")
        if not "References:" in answer_body: 
            if any(substr in answer_body for substr in ["\n * [", "\n * <", "\n* [", "\n* <"]):
                # there are already references in the answer, exit
                return answer_body
            else:
                # no references found, let's add some
                answer_body += "\n\nReferences:"
        # 2. references, priming the model to follow our prefered format
        prompt_extended = prompt + answer_body + "\n * <https://docs.nersc.gov"
        answer_references = self.base_generator(prompt_extended, stop_at="\n\n")
        # assemble the answer
        answer = answer_body + "\n * <https://docs.nersc.gov" + answer_references
        return answer
