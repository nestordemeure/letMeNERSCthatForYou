# Additional Scripts

## Scrontab Scripts

* `update_database.sh` causes the updating of the database (git pull of the repo then refreshing of the vector database),
* `api_worker.sh` starts an API worker,
* `api_worker_restarter.sh` restart the worker if it is not running.

The current [scrontab](https://docs.nersc.gov/jobs/workflow/scrontab/) file looks like this:

```shell
# Updates database for 20min tops everyday at 1am PST (9am UTC)
#SCRON --job-name=chatbot_database_update
#SCRON --account=nstaff
#SCRON --time=00:20:00
#SCRON -o /global/cfs/cdirs/nstaff/chatbot/letMeNERSCthatForYou/data/logs/update_database/output-%j.out
#SCRON --open-mode=append
0 9 * * * /global/cfs/cdirs/nstaff/chatbot/letMeNERSCthatForYou/scripts/update_database.sh

# Checks every 5 minutes if a worker is alive / pending, if not start one.
#SCRON --job-name=chatbot_worker_starter
#SCRON --account=nstaff
#SCRON --qos=cron
#SCRON --time=00:10:00
#SCRON --dependency=singleton
#SCRON -o /global/cfs/cdirs/nstaff/chatbot/letMeNERSCthatForYou/data/logs/api_worker/restarter_output-%j.out
#SCRON --open-mode=append
*/5 * * * * /global/cfs/cdirs/nstaff/chatbot/letMeNERSCthatForYou/scripts/api_worker_restarter.sh
```

You can use `scrontab -l` to vizualize it and `scrontab -e` to tweak it.

## Containers use

We use the [`asnaylor/lmntfy` image](https://hub.docker.com/r/asnaylor/lmntfy) based on the [vLLM image](https://docs.vllm.ai/en/latest/serving/deploying_with_docker.html) (see also our `Dockerfile`).

You can deploy it with [Shifter](https://docs.nersc.gov/development/containers/shifter/how-to-use/), running it interactively on a login node:

```sh
shifter --module=gpu,nccl-2.18 --image=asnaylor/lmntfy:v0.3 /bin/bash
```

You can also use it to preface a script call in order to use the container as your environment:

```sh
shifter --module=gpu,nccl-2.18 --image=asnaylor/lmntfy:v0.3 python3 chatbot.py
```

## Various Scripts

* `local_chatbot.sh` starts a chatbot running on the current node,
* `api_chatbot.sh` starts a chatbot client calling the API for answers,
* `dev_local_chatbot.sh` starts the dev chatbot script running on the current node,
* `dev_check_retrieval.sh` runs a few closest-chunks queries and display the result, to test our retrieval pipeline,
* `dev_cleverbot.sh` runs the best model available with no RAG, as a proof of concept.
