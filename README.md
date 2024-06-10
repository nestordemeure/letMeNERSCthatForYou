# Let Me NERSC that For You

This is a custom-made documentation Chatbot based on the [NERSC documentation](https://docs.nersc.gov/).

## Goals

This bot is not made to replace the documentation but rather to improve information discoverability.
Our goals are to:

* Being able to answer questions on NERSC with up-to-date, *sourced*, answers,
* run fully *open source* technologies *on-premise* giving us control, security, and privacy,
* serve this model to NERSC users in production with acceptable performance,
* make it fairly easy to switch the underlying models, embeddings and LLM, to be able to follow a rapidly evolving technology.

## Installation

* clone the repo,
* use the `environment.yml` file to install dependencies with `conda`
* clone the [NERSC doc repository](https://gitlab.com/NERSC/nersc.gitlab.io/-/tree/main/docs) into a folder.

## Usage

#### Basic use

Those scripts are meant to be run locally, mainly by developers of the project:

* `chatbot.py` this is a basic local question answering loop
* `chatbot_dev.py` is a more feature rich version of local loop, making it easy to run test questions and switch models around.
* `update_database.py` update the vector database (for a given llm, sentence embeder, and vector database)[^when]
* `token_counter.py` measure the size of questions and answers for a given tokenizer

On NERSC supercomputers, you might want to run `module load python cudatoolkit cudnn pytorch` before using those commands.

[^when]: This script is run once everyday (on a scron job).

#### Superfacility API use

Those scripts are meant to be user with the superfacility API:

* `api_client.py` this is a deonstration client, calling the chatbot via the superfacility API,
* `api_consumer.py` this is a worker, answering questions asked to the superfacility API on a loop

## TODO

In no particular order:

* move this code to the NERSC github,
* refresh prompt (moving chunk information within the prompt)

* move away from Whoosh (keyword search) to remove he need for a `search.initialize` method?

## Developers

<table width="100%">
  <tr>
    <td align="center">
      <a href="https://github.com/nestordemeure">
        <img src="https://github.com/nestordemeure.png" width="60" height="60" alt="Nestor Demeure" /><br>
        <a href="https://github.com/nestordemeure">Nestor Demeure</a><br>
        leading the effort and writing the glue code
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/ermalrrapaj">
        <img src="https://github.com/ermalrrapaj.png" width="60" height="60" alt="Ermal Rrapaj" /><br>
        <a href="https://github.com/ermalrrapaj">Ermal Rrapaj</a><br>
        finetuning and testing home-made models
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/gabor-lbl">
        <img src="https://github.com/gabor-lbl.png" width="60" height="60" alt="Gabor Torok" /><br>
        <a href="https://github.com/gabor-lbl">Gabor Torok</a><br>
        writing the superfacility API integration and web front-end
      </a>
    </td>
    <td align="center">
      <a href="https://github.com/asnaylor">
        <img src="https://github.com/asnaylor.png" width="60" height="60" alt="Andrew Naylor" /><br>
        <a href="https://github.com/asnaylor">Andrew Naylor</a><br>
        scaling the service to production throughputs
      </a>
    </td>
  </tr>
</table>
