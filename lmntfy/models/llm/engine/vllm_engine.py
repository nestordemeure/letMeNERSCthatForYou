import os
from typing import List
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.usage.usage_lib import UsageContext
from vllm.utils import random_uuid
import vllm
from . import LLMEngine

#----------------------------------------------------------------------------------------
# ENVIRONMENT

# avoids Ray duplicated logs (when using more than one GPU)
os.environ['RAY_DEDUP_LOGS'] = '0'

# deactivate the forced (!) exception on finish
vllm.engine.async_llm_engine._raise_exception_on_finish = lambda task, error_callback: None

#----------------------------------------------------------------------------------------
# INTERFACE

class VllmEngine(LLMEngine):
    """
    vLLM-based engine
    """
    def __init__(self, pretrained_model_name_or_path:str, device='cuda', nb_gpus=1, **engine_kwargs):
        # ensuring the device is a GPU
        if (device == 'cpu'):
            device = 'cuda'
            print("WARNING: switching device to GPU as VLLM currently only supports GPU")
        # load and starts the engine
        if (nb_gpus > 1): print(f"Setting up vLLM on {nb_gpus} GPUs, this might take some time.")
        engine_args = AsyncEngineArgs(model=pretrained_model_name_or_path, tensor_parallel_size=nb_gpus, device=device,
                                      disable_log_requests=True, disable_log_stats=False, **engine_kwargs)
        self.llm_engine = AsyncLLMEngine.from_engine_args(engine_args, start_engine_loop=True, usage_context=UsageContext.API_SERVER)
        # initializes the rest of the engine
        self.context_size = self.llm_engine.engine.model_config.max_model_len
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
        # note that the attribue might not exist due to early failure
        if hasattr(self, 'llm_engine') and (self.llm_engine._background_loop_unshielded is not None):
            self.llm_engine._background_loop_unshielded.cancel()
