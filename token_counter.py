#!/global/cfs/cdirs/nstaff/chatbot/conda/bin/python3
"""
This script can be used to determine an upper bound on the answer size for a given tokeniser.
Using preexisting answers.
"""
import json
import numpy as np
from pathlib import Path
import lmntfy

# Define your model here
models_folder = Path("/global/cfs/cdirs/nstaff/chatbot/models")
tokenizer = lmntfy.models.llm.Default(models_folder=models_folder, device='cpu').tokenizer
token_counter = tokenizer.count_tokens
print(f"Tokeniser type: {tokenizer.name}")

# Load JSON data
with open('./data/various/questions.json', 'r') as f:
    data = json.load(f)

for kind in ['answer', 'question']:
    print(f"*** {kind}:")
    # Extract answers and measure their sizes
    sizes = [token_counter(item[kind]) for item in data]

    # Convert to numpy array for easy calculation of statistics
    sizes = np.array(sizes)

    # Calculate and display statistics
    mean_size = np.mean(sizes)
    median_size = np.median(sizes)
    std_size = np.std(sizes)
    max_size = np.max(sizes)
    quantile_90 = np.percentile(sizes, 90)
    quantile_95 = np.percentile(sizes, 95)
    quantile_99 = np.percentile(sizes, 99)

    print(f"Mean size: {mean_size}")
    print(f"Median size: {median_size}")
    print(f"Standard deviation of size: {std_size}")
    print(f"Upper bound (mean+2std): {mean_size + 2*std_size}")
    print(f"Max size: {max_size}")
    print(f"90% quantile: {quantile_90}")
    print(f"95% quantile: {quantile_95}")
    print(f"99% quantile: {quantile_99}")

"""
Tiktoken (GPT35's tokeniser):
*** Answers:
Mean size: 96.20449718925671
Median size: 85.0
Standard deviation of size: 73.79027308535585
Upper bound (mean+2std): 243.78504335996843
Max size: 939
90% quantile: 183.0
95% quantile: 222.0
99% quantile: 345.0
(chunks of maximum size 819.2 tokens)
*** Questions:
Mean size: 13.172767020612117
Median size: 12.0
Standard deviation of size: 5.1105708284340805
Upper bound (mean+2std): 23.393908677480276
Max size: 130
90% quantile: 19.0
95% quantile: 22.0
99% quantile: 28.960000000000036

LlamaTokenizerFast (Llama2, Vicuna, Mistral, Zephyr):
*** answer:
Mean size: 107.95915053091818
Median size: 94.0
Standard deviation of size: 84.61246457852216
Upper bound (mean+2std): 277.1840796879625
Max size: 1096
90% quantile: 205.0
95% quantile: 252.0
99% quantile: 397.96000000000004
*** question:
Mean size: 15.17838850718301
Median size: 14.0
Standard deviation of size: 5.845451380160406
Upper bound (mean+2std): 26.869291267503822
Max size: 171
90% quantile: 22.0
95% quantile: 25.0
99% quantile: 32.0

Qwen2TokenizerFast (Qwen):
*** answer:
Mean size: 97.31930043722673
Median size: 86.0
Standard deviation of size: 75.4232263238064
Upper bound (mean+2std): 248.16575308483954
Max size: 954
90% quantile: 185.0
95% quantile: 225.0
99% quantile: 353.96000000000004
*** question:
Mean size: 13.236352279825109
Median size: 12.0
Standard deviation of size: 5.291229015598518
Upper bound (mean+2std): 23.818810311022148
Max size: 151
90% quantile: 19.0
95% quantile: 22.0
99% quantile: 29.0

Tokeniser type: PreTrainedTokenizerFast
*** answer:
Mean size: 97.19400374765772
Median size: 86.0
Standard deviation of size: 73.77911198753304
Upper bound (mean+2std): 244.7522277227238
Max size: 940
90% quantile: 184.0
95% quantile: 223.0
99% quantile: 346.0
*** question:
Mean size: 14.171767645221736
Median size: 13.0
Standard deviation of size: 5.110017833585836
Upper bound (mean+2std): 24.391803312393407
Max size: 131
90% quantile: 20.0
95% quantile: 23.0
99% quantile: 29.960000000000036
"""