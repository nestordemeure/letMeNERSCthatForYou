import asyncio
from typing import List
from torch import bfloat16
from transformers import AutoModelForCausalLM, AutoTokenizer
from .stopping_criteria import StopWordCriteria
from .. import LLMEngine

# Ensure that not more than one transformer model is currently running on the GPU
transformer_gpu_lock = asyncio.Lock()

class TransformerEngine(LLMEngine):
    """
    Hugginface's Transformer based engine.
    """
    def __init__(self, pretrained_model_name_or_path:str, device='cuda', model_kwargs:dict=dict()):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, 
                                                          device_map=device, torch_dtype=bfloat16,
                                                          **model_kwargs)
         # initializes the rest of the engine
        self.context_size = self.model.config.max_position_embeddings
        super().__init__(pretrained_model_name_or_path, self.context_size, device)

    async def generate(self, prompt:str, stopwords:List[str]=[], strip_stopword:bool=True, verbose:bool=False) -> str:
        """
        Query the model and get a response.

        Args:
            prompt (str): the text prompt
            stopwords (List[str]): the words on which to stop the generation, if any
            strip_stopword (bool): should we strip the stopword from our output (default to True)
            verbose (bool): should we print debug information? (defaults to False)

        Returns:
            str: The generated response from the model.
        """
        # used to stop on the stop words
        stopping_criteria = StopWordCriteria(tokenizer=self.tokenizer, prompts=[prompt], stop_words=stopwords)

        # tokenize the input text
        inputs_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        # runs the LLM, producing tokens for output=input+answer+stopword+?
        #output_tokens = self.model.generate(inputs_tokens, 
        #                                    max_length=self.context_size, 
        #                                    pad_token_id=self.tokenizer.eos_token_id,
        #                                    stopping_criteria=[stopping_criteria])
        # NOTE: we ensure that only one request is currently running on the GPU
        #       meanwhile, other CPU tasks can be done
        #       -> we could cut the code and have this engine be actualy synchronous (but this would be bad)
        async with transformer_gpu_lock:
            output_tokens = await asyncio.to_thread(self.model.generate, 
                                                    inputs_tokens, 
                                                    max_length=self.context_size, 
                                                    pad_token_id=self.tokenizer.eos_token_id,
                                                    stopping_criteria=[stopping_criteria])

        # extract answer text from output tokens, cutting prompt and stop words
        answer = stopping_criteria.extract_answers(output_tokens, strip_stopword=strip_stopword)[0]

        # debugging information
        if verbose: print(f"{prompt}\n{answer}")
        return answer