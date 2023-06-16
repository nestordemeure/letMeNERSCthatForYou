# Let Me NERSC that For You

This is a custom-made documentation Chatbot based on the [NERSC documentation](https://docs.nersc.gov/).

## Goals

This bot is not made to replace the documentation but rather to improve information discoverability.

* Being able to answer questions on NERSC with up-to-date, *sourced*, answers,
* make it fairly easy to switch the underlying models (embeddings and LLM) such that it can be run on-premise,
* have a user-facing web-based user interface.

## Installation

- clone the repo
- get an openai API key
- get the dependency in your environement (openai, tiktoken, faiss-cpu, sentence_transformers, rich)
- clone the [NERSC doc repository](https://gitlab.com/NERSC/nersc.gitlab.io/-/tree/main/docs) into a folder

## Usage

- put the openai API key in environement
- run lmntfy.py

## TODO

Model:
- have a Vicuna-based setup
- use the message model to answer followup questions?

Database:
- turn sources into url on loading chunks? (exploiting the local knowledge of the documentation folder to remove prefixes)
- improve urls by including a `#` link to the proper heading
- get a google search based retrieval set-up (nothing stored, instead we look up information online).

Web deployment:
- turn the code into an API?

Overall:
- cleanup readme