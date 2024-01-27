import re
from pathlib import Path
from typing import List, Dict
from . import LanguageModel, Answer

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

# system prompt used to pick an answer type
TRIAGE_PROMPT_SYSTEM = """
You are a member of the NERSC supercomputing center's support staff.
Your task is to act as a triage system for exchanges between NERSC supercomputing center users and support assistants. \
You must categorize the user's final message in the conversation into one of three categories: Technical Question, Out of Scope, or Small Talk.

### Categories and Formats:

1. **Technical Question**
   - **Action:** Identify technical inquiries needing documentation.
   - **Format:** Use `TECHNICAL_QUESTION(question:str)`.
   - **Example:** User asks about SSH connection → `TECHNICAL_QUESTION("How do I connect to NERSC using SSH?")`
   The restructured question should stand independently, crafted in such a way that the support team can comprehend and respond to it without referring back to the full conversation.

2. **Out of Scope**
   - **Action:** Recognize questions unrelated to NERSC's support scope.
   - **Format:** Use `OUT_OF_SCOPE`.
   - **Example:** User asks about general world facts → `OUT_OF_SCOPE`

3. **Small Talk**
   - **Action:** Identify casual, non-technical interactions.
   - **Format:** Use `SMALL_TALK(response:str)`.
   - **Example:** User says thanks → `SMALL_TALK("You're welcome!")`
   Your response should be conversational and appropriate, forwarding the user's sentiment in a manner that can be directly relayed to them.

### Conversation to be analyzed:
"""

# user prompt used to pick an answer type
TRIAGE_PROMPT_USER = """
Categorize the user's final message.
"""

#----------------------------------------------------------------------------------------
# MODEL

def comment_text(input_string:str) -> str:
    """Puts a "> " in front of each line of the given text."""
    # Split the input string into lines
    lines = input_string.split('\n')
    # Add '> ' in front of each line
    prefixed_lines = [f'> {line}' for line in lines]
    # Join the prefixed lines back into a single string
    result = '\n'.join(prefixed_lines)
    return result

class Vicuna(LanguageModel):
    def __init__(self, 
                 models_folder: Path,
                 model_name: str='vicuna-13b-v1.5',
                 device='cuda'):
        super().__init__(models_folder / model_name, device=device)
        self.tokenizer.chat_template = VICUNA_CHAT_TEMPLATE
        self.upper_answer_size = 450
        self.upper_question_size = 200

    def get_answer(self, question, chunks, verbose=False):
        """
        Method to get an answer given a question and some chunks passed for context.
        """
        # builds the messages
        nb_chunks = len(chunks)
        system_message = {"role": "system", "content": ANSWERING_PROMPT}
        context_messages = [{"role": "system", "content": f"\n{chunk.to_markdown()}", "relevancy": (nb_chunks-i)} for (i,chunk) in enumerate(chunks)]
        question_message = {"role": "user", "content": question}
        messages = [system_message] + context_messages + [question_message]
        # builds the prompt
        # NOTE: we prefix it with "Answer: " as the model has a tendancy to start with that anyway
        prompt = self.apply_chat_template(messages, nb_tokens_max=self.context_size-self.upper_answer_size) + "Answer: "
        # generates an answer
        # NOTE: we generate the answer in two part to ensure it follows our prefered format
        # 1. the text part
        answer_text = self.base_generator(prompt, stop_at="References:")
        print(f"DEBUGGING (answer_text):\n{answer_text}")
        if not "References:" in answer_text: return answer_text
        # 2. the references, priming them to follow our prefered format
        prompt_extended = prompt + answer_text + "\n* <"
        answer_references = self.base_generator(prompt_extended, stop_at="\n\n")
        print(f"DEBUGGING (answer_references):\n* <{answer_references}")
        # assemble the answer
        answer = answer_text + "\n* <" + answer_references
        return answer

    def triage(self, previous_messages:List[Dict], verbose=False) -> Answer:
        """
        Decides whether the message is:
        * out of scope,
        * a normal discussion (ie: "thank you!") that does not require a documentation call,
        * something that requires a documentation call
        """
        # builds the messages
        system_message = {"role": "system", "content": TRIAGE_PROMPT_SYSTEM}
        formatted_discussion = [{'role':'system', 'content': comment_text(f"\n**{message['role']}**: {message['content']}\n"), 'relevancy': i} for (i,message) in enumerate(previous_messages)]
        user_message = {"role": "user", "content": TRIAGE_PROMPT_USER}
        messages = [system_message] + formatted_discussion + [user_message]
        # builds the prompt
        prompt = self.apply_chat_template(messages, nb_tokens_max=self.context_size-self.upper_question_size)
        # generates an answer
        raw_answer = self.triage_generator(prompt)
        print(f"DEBUGGING (triage): {raw_answer}")
        # parse the raw answer
        if "OUT_OF_SCOPE" in raw_answer:
            return Answer.out_of_scope(raw=raw_answer)
        question_match = re.search(r'TECHNICAL\_QUESTION\("(.*?)"\)', raw_answer)
        if question_match:
            question_content = question_match.group(1)
            return Answer.question(question_content, raw=raw_answer)
        small_talk_match = re.search(r'SMALL\_TALK\("(.*?)"\)', raw_answer)
        if small_talk_match:
            small_talk_content = small_talk_match.group(1)
            return Answer.smallTalk(small_talk_content, raw=raw_answer)
        # default fallthought case (should be impossible with guided generation)
        question_content = previous_messages[-1]['content']
        return Answer.question(question_content, raw=raw_answer)

"""
TODO:
* make it so that "References:" is more likely to be used
"""