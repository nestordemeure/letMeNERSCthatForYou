from .. import LanguageModel, CHAT_PROMPT_SYSTEM, ANSWERING_PROMPT
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import random_uuid
import vllm
from pathlib import Path
from typing import List
import os

# avoids Ray duplicated logs (when using more than one GPU)
os.environ['RAY_DEDUP_LOGS'] = '0'

# deactivate the forced (!) exception on finish
vllm.engine.async_llm_engine._raise_exception_on_finish = lambda task, error_callback: None

#----------------------------------------------------------------------------------------
# INTERFACE

class VllmModel(LanguageModel):
    """
    model running on vllm
    """
    def __init__(self, pretrained_model_name_or_path:str, use_system_prompt=True, nb_gpus=1, device='cuda'):
        self.pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        self.device = device
        # load and starts the engine
        engine_args = AsyncEngineArgs(model=pretrained_model_name_or_path, tensor_parallel_size=nb_gpus, device=device,
                                      disable_log_requests=True, disable_log_stats=False)
        self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args, start_engine_loop=True, usage_context=UsageContext.LLM_CLASS)
        self.tokenizer = self.llm_engine.engine.get_tokenizer()
        # parameters of the model
        self.context_size = self.llm_engine.engine.model_config.max_model_len        
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
        sampling_params = SamplingParams(temperature=0, max_tokens=None, 
                                              stop=stopwords, include_stop_str_in_output=not strip_stopword)

        # generate an async iterator
        results_generator = self.llm_engine.generate(prompt, sampling_params, request_id=random_uuid())

        # Non-streaming case, gets to the end of the async iterator
        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        # extract answer text
        answer = final_output.outputs[0].text

        # debugging information
        if verbose: print(f"{prompt}\n{answer}")
        return answer

    def __del__(self):
        """gets rid of the (while True) engine_loop task on deletion"""
        self.llm_engine._background_loop_unshielded.cancel()

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
