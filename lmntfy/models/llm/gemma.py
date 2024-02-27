import torch
from pathlib import Path
from . import LanguageModel

# Jinja chat template
# found [here](https://github.com/chujiezheng/chat_templates/blob/main/chat_templates/gemma-it.jinja)
GEMMA_CHAT_TEMPLATE = """
{% if messages[0]['role'] == 'system' %}
    {% set loop_messages = messages[1:] %}
    {% set system_message = messages[0]['content'].strip() + '\n\n' %}
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
    {% if (message['role'] == 'assistant') %}
        {% set role = 'model' %}
    {% else %}
        {% set role = message['role'] %}
    {% endif %}
    {{ '<start_of_turn>' + role + '\n' + message['content'].strip() + '<end_of_turn>\n' }}
{% endfor %}
{% if add_generation_prompt %}
    {{'<start_of_turn>model\n'}}
{% endif %}
"""

class Gemma(LanguageModel):
    def __init__(self, 
                 models_folder: Path,
                 model_name: str='gemma-7b-it',
                 device='cuda'):
        super().__init__(models_folder / model_name, device=device,
                         model_kwargs={'torch_dtype':torch.bfloat16, 'attn_implementation':'flash_attention_2'})
        self.tokenizer.chat_template = GEMMA_CHAT_TEMPLATE
        self.upper_answer_size = 450 # TODO update
        self.upper_question_size = 200 # TODO update

    def apply_chat_template(self, messages, nb_tokens_max:int=None) -> str:
        """
        Converts the system message into a user message as the model does not accept system messages.
        """
        # merge system messages (in case there is more than one)
        merged_messages = self._clean_messages(messages)
        # shorcut if we built a prompt with no systen message
        if merged_messages[0]['role'] != 'system': return messages
        system_message = merged_messages[0]
        chat_messages = merged_messages[1:]

        # turn the system message into a user message followed by ok as system messages are not allowed by the model
        system_prompt_message = {'role':'user', 'content':system_message['content']}
        okay_message = {'role':'assistant', 'content':"Okay! From now on I will talk as a member of the NERSC supercomputing center's support staff discussing with a user."}
        no_system_messages = [system_prompt_message, okay_message] + chat_messages

        # run the main class's apply_chat function
        return super().apply_chat_template(messages=no_system_messages, nb_tokens_max=nb_tokens_max)