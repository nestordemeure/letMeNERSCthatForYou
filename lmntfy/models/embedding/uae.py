import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from . import Embedding


class UAEEmbedding(Embedding):
    """
    https://huggingface.co/WhereIsAI/UAE-Large-V1
    """
    def __init__(self, 
                 models_folder,
                 name='UAE-Large-V1', 
                 embedding_length=1024,
                 max_input_tokens=512,
                 normalized=False, # TODO to check
                 device=None):
        super().__init__(models_folder, name, embedding_length, max_input_tokens, normalized, device)
        # loads the model
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(self.pretrained_model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation='flash_attention_2').to(device)
        # task specific prompts
        # see: https://github.com/SeanLee97/AnglE/blob/main/angle_emb/angle.py
        self.retrieval_prompt = 'Represent this sentence for searching relevant passages: {text}'

    def _embed(self, text, is_query=False):
        """
        Computes the embedding of a given text
        """
        # format input for the task (retrieval)
        text_formated = self.retrieval_prompt.format(text=text)
        # turns the text into tokens
        tokens = self.tokenizer([text_formated], return_tensors='pt')
        # move them to device
        for k, v in tokens.items():
            tokens[k] = v.to(self.device)
        # computes the embedding
        embedding = self.model(output_hidden_states=True, **tokens).hidden_states[-1][:, -1].float().detach().cpu().numpy()
        return embedding

    def count_tokens(self, text):
        """
        Counts the number of tokens used to represent the given text
        """
        tokens = self.tokenizer.encode(text, return_tensors='pt')
        token_number = tokens.size(-1)
        return token_number