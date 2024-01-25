from pathlib import Path
from typing import List, Dict
from ...question_answering import Answer
from . import LanguageModel

#----------------------------------------------------------------------------------------
# PROMPTS

# Jinja chat template for Vicuna
# found [here](https://github.com/chujiezheng/chat_templates/blob/main/chat_templates/vicuna.jinja)
VICUNA_CHAT_TEMPLATE = """
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

# prompt to answer a question
# NOTE: 
# * we do a single shot prompt (with an example answer) to ensure proper formating of the answer at the price of a few tokens
# * note that the end of the prompt is ready to accomodate some chunks of information
ANSWERING_PROMPT="""
You are a member of the NERSC supercomputing center's support staff.
Generate a comprehensive and informative answer for a given question solely based on the provided information (URL and Extract).
You must only use information from the provided search results.
Use an unbiased and journalistic tone.
Combine search results together into a coherent answer.
Only cite the most relevant results that answer the question accurately.
Try and be careful not to go off-topics.
After providing the answer, list the URLs of the information sources you used, sorted from most to least relevant. Include ONLY the URLs that are directly relevant to the answer.

### Example Answer Format:

Question: How can I optimize my code for CPU at NERSC?

Answer: To optimize your code for CPU usage at NERSC, it's crucial to focus on vectorization and parallelization. Vectorization allows your code to process multiple data points with a single instruction, effectively reducing the time your code takes to run through large datasets. Parallelization, on the other hand, involves dividing your code into multiple tasks that can be processed simultaneously, maximizing the use of available CPU resources. Combining these two strategies can lead to significant improvements in your code's performance on NERSC systems.

References:
* <https://nersc.example.com/documentation/cpu-optimization>
* <https://nersc.example.com/documentation/parallel-computing>

### Information Sources:
"""

# prompt to summarize a conversation into its latest question
# NOTE: 
# * we do a single shot prompt (with an example answer) to ensure proper formating of the answer at the price of a few tokens
# * note that the end of the prompt is ready to accomodate the conversation
QUESTION_EXTRACTION_PROMPT_SYSTEM = """
Your task is to function as a question extraction system.
Your input will be the concluding part of an exchange between a NERSC supercomputing center user and a support assistant.
Your primary objective is to distill and rephrase the final inquiry posed by the user.
The restructured question should stand independently, crafted in such a way that the support team can comprehend and respond to it without referring back to the full conversation.
The refined question should be enclosed within a code block.

### Example Answer Format:

```
How can I optimize my code to efficiently utilize more nodes in the NERSC supercomputing center?
```

### Conversation:
"""
QUESTION_EXTRACTION_PROMPT_USER = """
Extract the user's last message from the conversation.
"""

# system prompt used to pick an answer type
TRIAGE_PROMPT_SYSTEM = """
Your task is to function as triage.
Your input will be the concluding part of an exchange between a NERSC supercomputing center user and a support assistant.
Your primary objective is to decide if the final inquiry posed by the user is a technical question, out of scope, or small talk.

### Answer Format:

#### Technical Question

This is the most common case, a technical question that requires consulting the documentation in order to produce a proper answer.

For example, "Can I use SSH to do this?" (in a discussion about connecting to NERSC).

You answer should be of the form `QUESTION(question:str)`, for example: `QUESTION(Can I connect to NERSC using SSH?)`.
The restructured question should stand independently, crafted in such a way that the support team can comprehend and respond to it without referring back to the full conversation.

#### Out of scope

For example, asking about facts (such as world facts) that are not covered by NERSC's documentation.

You answer should be `OUT_OF_SCOPE`.

#### Small Talk

For example, thanking you for your help.

You answer should be of the form `SMALL_TALK(answer:str)`, for example: `SMALL_TALK(You are welcome!)`.
Your answer will be forwarded to the user.

### Conversation:
"""

# user prompt used to pick an answer type
TRIAGE_PROMPT_USER = """
Pick an answer type for the user's last message.
"""

#----------------------------------------------------------------------------------------
# MODEL

class Vicuna(LanguageModel):
    def __init__(self, 
                 models_folder: Path,
                 model_name: str='vicuna-13b-v1.5',
                 device='cuda'):
        super().__init__(models_folder / model_name, device)
        self.tokenizer.chat_template = VICUNA_CHAT_TEMPLATE
        self.upper_answer_size = 450
        self.upper_question_size = 200

    def get_answer(self, question, chunks, verbose=False):
        """
        Method to get an answer given a question and some chunks passed for context.
        """
        nb_chunks = len(chunks)
        # builds the prompt
        system_message = {"role": "system", "content": ANSWERING_PROMPT}
        context_messages = [{"role": "system", "content": f"\n{chunk.to_markdown()}", "relevancy": (nb_chunks-i)} for (i,chunk) in enumerate(chunks)]
        question_message = {"role": "user", "content": question}
        messages = [system_message] + context_messages + [question_message]
        # runs the query
        answer = self.query(messages, expected_answer_size=self.upper_answer_size, verbose=verbose)
        # remove potential prefix
        answer = answer.split("Answer: ", 1)[1] if ("Answer: " in answer) and answer.startswith(("Answer: ", "Question: ")) else answer
        # remove potential suffix
        answer = answer.split("URLs:")[0].strip() if ("URLs:" in answer) and ("References:" in answer) and (answer.index("URLs:") > answer.index("References:")) else answer
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
        formatted_discussion = [{'role':'system', 'content': f"\n**{message['role']}**: {message['content']}\n", 'relevancy': i} for (i,message) in enumerate(previous_messages)]
        user_message = {"role": "user", "content": QUESTION_EXTRACTION_PROMPT_USER}
        messages = [system_message] + formatted_discussion + [user_message]
        # queries the system
        question = self.query(messages, expected_answer_size=self.upper_question_size, verbose=verbose)
        # extract question from code block
        question = question.split('```')[-2].strip() if (question.count('```') >= 2) else question
        # deal with empty string failure case
        question = previous_messages[-1]['content'] if (len(question) == 0) else question
        return question

    def triage(self, messages:List[Dict], verbose=False) -> Answer:
        """
        Decides whether the message is:
        * out of scope,
        * a normal discussion (ie: "thank you!") that does not require a documentation call,
        * something that requires a documentation call
        """
        # builds the prompt
        system_message = {"role": "system", "content": TRIAGE_PROMPT_SYSTEM}
        formatted_discussion = [{'role':'system', 'content': f"\n**{message['role']}**: {message['content']}\n", 'relevancy': i} for (i,message) in enumerate(messages)]
        user_message = {"role": "user", "content": TRIAGE_PROMPT_USER}
        messages = [system_message] + formatted_discussion + [user_message]
        # queries the system
        raw_answer = self.query(messages, expected_answer_size=self.upper_question_size, verbose=verbose)
        print(f"DEBUGGING: <{raw_answer}>")
        # decide on the type of answer
        answer = Answer.is_out_of_scope()
        return answer
