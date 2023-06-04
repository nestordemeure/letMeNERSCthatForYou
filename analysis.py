"""
This script can be used to determine an upper bound on the answer size for a given tokeniser.
Using preexisting answers.
"""
from lmntfy.models.llm import GPT35
import numpy as np
import json

# Define your token_counter function here
token_counter = GPT35().token_counter

# Load JSON data
with open('./data/questions.json', 'r') as f:
    data = json.load(f)

# Extract answers and measure their sizes
answer_sizes = [token_counter(item['answer']) for item in data]

# Convert to numpy array for easy calculation of statistics
answer_sizes = np.array(answer_sizes)

# Calculate and display statistics
mean_size = np.mean(answer_sizes)
median_size = np.median(answer_sizes)
std_size = np.std(answer_sizes)
max_size = np.max(answer_sizes)
quantile_90 = np.percentile(answer_sizes, 90)
quantile_95 = np.percentile(answer_sizes, 95)
quantile_99 = np.percentile(answer_sizes, 99)

print(f"Mean size: {mean_size}")
print(f"Median size: {median_size}")
print(f"Standard deviation of size: {std_size}")
print(f"Upper bound (mean+2std): {mean_size + 2*std_size}")
print(f"Max size: {max_size}")
print(f"90% quantile: {quantile_90}")
print(f"95% quantile: {quantile_95}")
print(f"99% quantile: {quantile_99}")

"""
GPT35's tokeniser:
Mean size: 96.20449718925671
Median size: 85.0
Standard deviation of size: 73.79027308535585
Upper bound (mean+2std): 243.78504335996843
Max size: 939
90% quantile: 183.0
95% quantile: 222.0
99% quantile: 345.0
(chunks of maximum size 819.2 tokens)
"""