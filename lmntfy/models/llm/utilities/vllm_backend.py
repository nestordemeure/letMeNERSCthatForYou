import torch
from pathlib import Path
from typing import List
from .. import LanguageModel, CHAT_PROMPT_SYSTEM, ANSWERING_PROMPT
import vllm

#----------------------------------------------------------------------------------------
# INTERFACE

class VllmModel(LanguageModel):
    """
    model running on vllm
    """
    def __init__(self, pretrained_model_name_or_path:str, use_system_prompt=True, nb_gpus=1, device='cuda'):
        self.pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        self.device = device
        self.model = vllm.LLM(model=pretrained_model_name_or_path, tensor_parallel_size=nb_gpus, device=device)
        self.tokenizer = self.model.get_tokenizer()
        self.context_size = self.model.llm_engine.model_config.max_model_len
        self.upper_answer_size = None # needs to be filled per tokenizer
        self.upper_question_size = None # needs to be filled per tokenizer
        self.use_system_prompt = use_system_prompt
        # prompts
        self.CHAT_PROMPT_SYSTEM = CHAT_PROMPT_SYSTEM
        self.ANSWERING_PROMPT = ANSWERING_PROMPT

    async def generate(self, prompt:str, stopwords:List[str]=[], strip_stopword:bool=True, verbose:bool=False) -> str:
        """
        Query the model and get a response.
        NOTE: this function is written to deal with a single piece of text, not a batch

        Args:
            prompt (str): the text prompt
            stopwords (List[str]): the words on which to stop the generation, if any
            strip_stopword (bool): should we strip the stopword from our output (default to True)
            verbose (bool): should we print debug information? (defaults to False)

        Returns:
            str: The generated response from the model.
        """
        # defines the sampling parameters with our stopping criteria
        sampling_params = vllm.SamplingParams(temperature=0, max_tokens=None, 
                                              stop=stopwords, include_stop_str_in_output=not strip_stopword)

        # generate a List[CompletionOutput]
        outputs = self.model.generate([prompt], sampling_params, use_tqdm=False)
        
        # extract answer text
        answer = outputs[0].outputs[0].text

        # debugging information
        if verbose: print(f"{prompt}\n{answer}")
        return answer

"""
TODO:
use the asyncengine?
-> run a test to see if the current model can deal concurently with async request
-> if not, add async method
"""

#----------------------------------------------------------------------------------------
# MODELS

class MistralVllm(VllmModel):
    def __init__(self, 
                 models_folder: Path,
                 model_name: str='Mistral-7B-Instruct-v0.2',
                 use_system_prompt=False,
                 device='cuda'):
        super().__init__(models_folder / model_name, use_system_prompt=use_system_prompt, device=device)
        self.upper_answer_size = 450
        self.upper_question_size = 200
