import torch
from pathlib import Path
from . import LanguageModel

# Jinja chat template
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

class Llama2(LanguageModel):
    def __init__(self, 
                 models_folder: Path,
                 model_name: str='Llama-2-13b-chat-hf',
                 device='cuda'):
        super().__init__(models_folder / model_name, device=device,
                         model_kwargs={'torch_dtype':torch.bfloat16, 'attn_implementation':'flash_attention_2'})
        self.tokenizer.chat_template = LLAMA2_CHAT_TEMPLATE
        self.upper_answer_size = 450
        self.upper_question_size = 200
