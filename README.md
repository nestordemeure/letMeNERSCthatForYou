# Let Me NERSC that For You

This is a custom-made documentation Chatbot based on the [NERSC documentation](https://docs.nersc.gov/).

## Goals

This bot is not made to replace the documentation but rather to improve information discoverability.

* Being able to answer questions on NERSC with up-to-date, *sourced*, answers,
* make it fairly easy to switch the underlying models (embeddings and LLM) such that it can be run on-premise,
* have a user-facing web-based user interface.

## Installation

**TODO**
- clone the repo
- get an openai API key
- get the dependency in your environement (openai, tiktoken, faiss-cpu, sentence_transformers)

## Usage

**TODO:**
- put the openai API key in environement
- run lmntfy.py

## TODO

Q&A:
- improve source answer setup
  - simplify number promtp to ask number in just `[number]` format?
  - ask in prompt for them to be put at the end of the message?
- harden model against error (input size, model failure, etc: if retying is not enough then we should return an error message to the user)
  - log any error?

Model:
- have a Vicuna-based setup

Database:
- merge Database and VectorDatabase classes
- get a google search based retrieval set-up (nothing stored, instead we look up information online).

UI:
- put the command line UI into its own file
- web UI
  - turn the code into an API?
