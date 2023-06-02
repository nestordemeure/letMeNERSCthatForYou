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
- get an api key
- get the dependency in your environement

## Usage

**TODO:**
- put the API key in environement
- run lmntfy.py

## TODO

#### Global

Q&A:
- have a single question answer set up
- decide on 10 demo questions
- have a sourced answer setup
- have a chat (back and forth) set-up?

Model:
- get a chatgpt3.5 based answering setup
- have a vicuna based alternative setup

Embeddings:
- get the doc embedded with the openai model
- get SBERT embeddings working as a viable alternative

Database:
- get a naive version running
- get a FAISS based version running
- get a google search based retrieval set-up (nothing stored, instead we look up information online).
- add input preprocessing to turn question into something closer to the target

UI:
- basic automated Q&A function for tests
- command line UI
- web UI

Further points:
- use API key in environement
- no need for rate limiting
- yes need for number of tokens limiting

#### Details

move embedding normalisation into the embedding class rather than the database