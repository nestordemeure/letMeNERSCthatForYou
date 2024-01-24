import sys
from pathlib import Path
from . import LanguageModel

#----------------------------------------------------------------------------------------
# PROMPTS

# prompt to answer a question
ANSWERING_PROMPT="You are a member of the NERSC supercomputing center's support staff. \
Generate a comprehensive and informative answer for a given question solely based on the provided information (URL and Extract). \
You must only use information from the provided search results. \
Use an unbiased and journalistic tone. \
Combine search results together into a coherent answer. \
Only cite the most relevant results that answer the question accurately. \
Try and be careful not to go off-topics."

# prompt to summarize a conversation into its latest question
QUESTION_EXTRACTION_PROMPT_SYSTEM="You are a question extraction system. \
You will be provided the last messages of a conversation between a user of the NERSC supercomputing center and an assistant from its support, ending on a question by the user. \
Your task is to summarize the user's last message."
QUESTION_EXTRACTION_PROMPT_USER="Reframe my last question so that I can forward it to support without the rest of this conversation."

#----------------------------------------------------------------------------------------
# MODEL

class Vicuna(LanguageModel):
    def __init__(self, 
                 models_folder: Path,
                 model_name: str='vicuna-13b-v1.5',
                 device='cuda'):
        super().__init__(models_folder / model_name, device)
        self.upper_answer_size = 450
        self.upper_question_size = 200
        # Vicuna comes without a template
        # found [here](https://github.com/chujiezheng/chat_templates/blob/main/chat_templates/vicuna.jinja)
        self.tokenizer.chat_template = """
{% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content'].strip() + '\n\n' %}
{% else %}
    {% set loop_messages = messages %}
    {% set system_message = '' %}
{% endif %}
{{ bos_token + system_message }}
{% for message in loop_messages %}
    {% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}
        {{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}
    {% endif %}
    {% if message['role'] == 'user' %}
        {{ 'USER: ' + message['content'].strip() + '\n' }}
    {% elif message['role'] == 'assistant' %}
        {{ 'ASSISTANT: ' + message['content'].strip() + eos_token + '\n' }}
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}
    {{ 'ASSISTANT:' }}
{% endif %}
"""

    def get_answer(self, question, chunks, verbose=False):
        """
        Method to get an answer given a question and some chunks passed for context.
        """
        nb_chunks = len(chunks)
        # builds the prompt
        system_message = {"role": "system", "content": ANSWERING_PROMPT}
        context_messages = [{"role": "system", "content": f"\n\n{str(chunk)}", "relevancy": (nb_chunks-i)} for (i,chunk) in enumerate(chunks)]
        question_message = {"role": "user", "content": question}
        messages = [system_message] + context_messages + [question_message]
        # runs the query
        answer = self.query(messages, expected_answer_size=self.upper_answer_size, verbose=verbose)
        return answer

    def extract_question(self, previous_messages, verbose=False):
        """
        Extracts the latest question given a list of messages.
        Message are expected to be dictionnaries with a 'role' ('user' or 'assistant') and 'content' field.
        the question returned will be a string.
        """
        # shortcut for single (first) questions
        if len(previous_messages) == 1:
            return previous_messages[0]['content']
        # builds the prompt
        system_message = {"role": "system", "content": QUESTION_EXTRACTION_PROMPT_SYSTEM}
        # the conversation is put inside the system prompt
        discussion_introduction = {'role':'system', 'content': "\n\nConversation:\n"}
        formatted_discussion = [{'role':'system', 'content': f"{message['role']}:{message['content']}\n", 'relevancy': i} for (i,message) in enumerate(previous_messages)]
        user_message = {"role": "user", "content": QUESTION_EXTRACTION_PROMPT_USER}
        messages = [system_message, discussion_introduction] + formatted_discussion + [user_message]
        # queries the system
        question = self.query(messages, expected_answer_size=self.upper_question_size, verbose=verbose)
        # combine it with the last nessage so that we cover all of our bases
        last_message = previous_messages[-1]['content']
        question = f"{last_message} ({question})"
        return question
