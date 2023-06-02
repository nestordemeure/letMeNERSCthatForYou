import tiktoken
import openai
from . import LanguageModel


class GPT35(LanguageModel):
    def __init__(self, 
                 model_name='gpt-3.5-turbo', 
                 context_size=4096):
        super().__init__(model_name, context_size)
        self.tokenizer = tiktoken.encoding_for_model(model_name)
        self.model_tokens_per_message = 4
        self.model_tokens_per_name = -1

    def _token_counter_messages(self, messages):
        """
        GPT-3.5-turbo specific token counting implementation for list of messages.
        """
        total_tokens = 0
        for message in messages:
            total_tokens += self.model_tokens_per_message
            for key, value in message.items():
                total_tokens += self.token_counter(value)
                if key == "name":
                    total_tokens += self.model_tokens_per_name
        total_tokens += 2
        return total_tokens

    def token_counter(self, text):
        """
        GPT-3.5-turbo specific token counting implementation.
        """
        encoded_text = self.tokenizer.encode(text)
        return len(encoded_text)

    def query(self, text):
        """
        GPT-3.5-turbo specific model query and response.
        """
        if self.is_input_too_long(text):
            raise ValueError("Input text is too long for the model context size.")
        
        response = openai.ChatCompletion.create(
          model=self.model_name,
          messages=[{"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": text}])
        
        return response.choices[0].message['content']
