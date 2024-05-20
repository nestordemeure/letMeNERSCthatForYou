# transformer engine
# NONE

# vLLM engine
from .llama2 import Llama2 # hallucinate often
from .llama2 import Vicuna #good but I have had some cut-offs problems
from .llama2 import CodeLlama # good answers but does not care much for the provided doc
from .mistral import Mistral # good at answering, not at picking references
from .mistral import Zephyr # good but can miss some information from the doc provided
from .mistral import OpenChat # answers somewhat in league with mistral
from .mistral import Snorkel # good answers but high hallucinations
from .mistral import Starling # a bit verbose but very good
from .mistral import StarlingCode # not as good as base (understandable as this one is for code writing only)
from .mistral import Mixtral # too heavy to serve on a single GPU
from .gemma import Gemma # tends to answer not quite the question asked (TODO to be reevaluated)
from .qwen import Qwen # really nice (feels competitive with mistral)
from .llama3 import Llama3 # very good, if a bit litteral in its understanding of queries

# default model
Default = Llama3
