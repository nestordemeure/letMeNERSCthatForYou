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
You must only use information from the provided search results.
Use an unbiased and journalistic tone.
Combine search results together into a coherent answer.
Only cite the most relevant results that answer the question accurately.
Try and be careful not to go off-topics.
After providing the answer, list the URLs of the information sources you used in a `References:` section, sorted from most to least relevant. Include ONLY the URLs that are directly relevant to the answer.

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
Your task is to act as a triage system for exchanges between NERSC supercomputing center users and support assistants.
Your primary role is to accurately categorize the **last message** of the user in the conversation into one of three distinct categories: Technical Question, Out of Scope, or Small Talk. It's crucial to focus on the last message and its context to ensure accurate classification.

### Categories and Formats:

1. **Technical Question**
   - **Action:** Identify the last message that specifically requires technical guidance or documentation related to NERSC services or systems.
   - **Criteria:** The message should be directly seeking information or assistance related to the technical aspects, functionalities, or procedures of NERSC services, and it must be the final message in the conversation.
   - **Format:** Use `TECHNICAL_QUESTION(question:str)`.
   - **Example:** User asks about SSH connection → `TECHNICAL_QUESTION("How do I connect to NERSC using SSH?")`
   - **Note:** Ensure the restructured question is clear, self-contained, and accurately represents the last message, enabling the support team to understand and address it without needing additional context.

2. **Out of Scope**
   - **Action:** Recognize and flag the last message that inquires about topics beyond the support and operational scope of NERSC.
   - **Criteria:** The message discusses subjects not related to NERSC's services, technical support, or operational procedures, and it must be the final message in the conversation.
   - **Format:** Use `OUT_OF_SCOPE`.
   - **Example:** User asks about general world facts or non-NERSC topics → `OUT_OF_SCOPE`

3. **Small Talk**
   - **Action:** Identify the last message that is conversational and non-technical in nature, typically involving greetings, gratitude, or general well-being.
   - **Criteria:** The message does not seek technical support or specific information but rather engages in a casual, polite, or social manner, and it must be the final message in the conversation.
   - **Format:** Use `SMALL_TALK(response:str)`.
   - **Example:** User says thanks → `SMALL_TALK("You're welcome!")`
   - **Note:** Your response should be warm and conversational, reflecting the sentiment expressed by the user, suitable for direct communication.

### Guidelines for Classification:
- **Focus on the Last Message:** Ensure that the categorization is based solely on the last message of the conversation. Avoid considering previous messages unless they provide essential context for understanding the last message.
- **Context Matters:** Evaluate the last message not just by its content but also in the context of the entire conversation.
- **Accuracy is Key:** Take the time to carefully read and understand the user's final message to ensure your classification aligns with the user's intent and the nature of their inquiry.

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
        if not "References:" in answer_text: 
            if ("\n * [" in answer_text) or ("\n * <" in answer_text):
                # there is already a list of links
                return answer_text
            else:
                # no references found, let's add some
                answer_text += "\n\nReferences:"
        # 2. the references, priming them to follow our prefered format
        prompt_extended = prompt + answer_text + "\n* <"
        answer_references = self.base_generator(prompt_extended, stop_at="\n\n")
        # assemble the answer
        answer = answer_text + "\n* <" + answer_references
        return answer

    def old_triage(self, previous_messages:List[Dict], verbose=False) -> Answer:
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

    def triage(self, previous_messages:List[Dict], verbose=False) -> Answer:
        """
        Decides whether the message is:
        * out of scope,
        * a normal discussion (ie: "thank you!") that does not require a documentation call,
        * something that requires a documentation call
        """
        # builds the messages
        system_message = {"role": "system", "content": CHAT_PROMPT_SYSTEM}
        formatted_discussion = [{**message, 'relevancy': i} for (i,message) in enumerate(previous_messages)]
        messages = [system_message] + formatted_discussion
        # builds the base prompt
        prompt = self.apply_chat_template(messages, nb_tokens_max=self.context_size-self.upper_answer_size)
        # requires doc?
        prompt_use_rag = prompt + '[Note to self: is this last message a technical question than can be answered by the NERSC doc? '
        use_rag = self.yesno_generator(prompt_use_rag) == 'Yes'
        print(f"DEBUGGING: use_rag:{use_rag}")
        if use_rag:
            # question extraction
            prompt_question_extraction = prompt + 'If I understand you clearly, your question is: "'
            question = self.base_generator(prompt_question_extraction, stop_at='"')[:-1]
            print(f"DEBUGGING: question:'{question}'")
            # concatenates it with the user's last message to reduce accidental question loss
            users_last_message = previous_messages[-1]['content']
            full_question = f"{users_last_message} ({question})"
            return Answer.question(full_question, raw=question)
        else:
            # is it a (non-technical) question?
            prompt_is_question = prompt + '[Note to self: is this last message a question? '
            is_question = self.yesno_generator(prompt_is_question) == 'Yes'
            print(f"DEBUGGING: out_of_scope:{is_question}")
            if is_question:
                # non-technical question
                return Answer.out_of_scope()
            else:
                # answer directly
                answer = self.base_generator(prompt)
                return Answer.smallTalk(answer, raw=answer)
