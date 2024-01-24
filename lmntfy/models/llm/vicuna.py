import sys
from pathlib import Path
from . import LanguageModel, keep_references_only

#----------------------------------------------------------------------------------------
# PROMPTS

# prompt to answer a question
ANSWERING_PROMPT="A chat between a curious user and an artificial intelligence assistant. \
The assistant gives helpful, detailed, and polite answers to the user's questions. \
The assistant generates a comprehensive and informative answer for a given question solely based on the provided information (URL and Extract). \
The assistant combines search results together into a coherent answer. \
The assitant only cites the most relevant results that answer the question accurately. \
The assistant answer the users's question based only on the following information:"
# this prompt is meant to be followed by a list of chunks

# prompt to answer a question
ANSWERING_PROMPT="You are a member of the NERSC supercomputing center's support staff. \
Generate a comprehensive and informative answer for a given question solely based on the provided information (URL and Extract). \
You must only use information from the provided search results. \
Use an unbiased and journalistic tone. \
Combine search results together into a coherent answer. \
Only cite the most relevant results that answer the question accurately. \
Try and be careful not to go off-topics."

# prompt to get references for an answer
REFERENCE_PROMPT="Produce a bullet list of the urls you found useful to answer the question, \
sorted from most relevant to least relevant. \
Keep the bullet list short, keeping *only* the relevant urls (there are rarely more than three relevant urls)."

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
                 model_name: str='vicuna-13b-v1.3',
                 device='cuda'):
        super().__init__(models_folder / model_name, device)
        self.upper_answer_size = 450
        self.upper_question_size = 200
        # Vicuna comes without a template and the default is not a fit.
        # follows the template given [here](https://github.com/lm-sys/FastChat/blob/main/docs/vicuna_weights_version.md#prompt-template)
        self.tokenizer.chat_template = """
{% for message in messages %}
    {% if message.role == 'system' %}
        {{ message.content.strip() }}\n\n
    {% elif message.role == 'USER' %}
        USER: {{ message.content.strip() }}\n
    {% elif message.role == 'ASSISTANT' %}
        ASSISTANT: {{ message.content.strip() }}{{ sep2 }}\n
    {% endif %}
{% endfor %}
{% if add_generation_prompt %}
    ASSISTANT:
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
        # adds sources at the end of the query
        # TODO answer = self.add_references(question, answer, chunks, verbose=verbose)
        # returns
        return answer
        
    # def add_references(self, question, answer, chunks, verbose=False):
    #     """
    #     Adds references to an answer.
    #     """
    #     # builds the prompt
    #     system_message  = {"role": "system", "content": "URLs available:"}
    #     context_messages = [{"role": "system", "content": f"\n\n{str(chunk)}"} for chunk in chunks]
    #     question_message = {"role": "user", "content": question}
    #     answer_message = {"role": "assistant", "content": answer}
    #     reference_message = {"role": "user", "content": REFERENCE_PROMPT}
    #     messages = [system_message] + context_messages + [question_message, answer_message, reference_message]
    #     # runs the query
    #     prompt = self.messages_to_prompt(messages)
    #     references = self.query(prompt, verbose=verbose)
    #     # remove any irrelevant lines
    #     references = keep_references_only(references)
    #     # updates the answer
    #     answer = f"{answer}\n\nReferences:\n{references}"
    #     return answer

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
        # remove an eventual prefix
        # TODO if question.startswith('user: '): question = question[6:]
        # combine it with the last nessage so that we cover all of our bases
        last_message = previous_messages[-1]['content']
        question = f"{last_message} ({question})"
        return question
