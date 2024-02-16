# Additional Scripts

## Scrontab Scripts

* `update_database.sh` causes the updating of the database (git pull of the repo then refreshing of the vector database),
* `api_worker.sh` starts an API worker,
* `api_worker_restarter.sh` restart the worker if it is not running.

The current [scrontab](https://docs.nersc.gov/jobs/workflow/scrontab/) file looks like this:

```shell
# Updates database for 10min everyday at 1am PST (9am UTC)
#SCRON --job-name=chatbot_database_update
#SCRON --account=nstaff
#SCRON --time=00:10:00
#SCRON -o /global/cfs/cdirs/nstaff/chatbot/letMeNERSCthatForYou/data/logs/update_database/output-%j.out
#SCRON --open-mode=append
0 9 * * * /global/cfs/cdirs/nstaff/chatbot/letMeNERSCthatForYou/scripts/update_database.sh

# Starts worker for 24h at 1:15am PST (9:15am UTC)
#SCRON --job-name=singleton_chatbot_worker
#SCRON --account=nstaff
#SCRON --time=24:00:00
#SCRON --dependency=singleton
#SCRON -o /global/cfs/cdirs/nstaff/chatbot/letMeNERSCthatForYou/data/logs/api_worker/worker_output-%j.out
#SCRON --open-mode=append
15 9 * * * /global/cfs/cdirs/nstaff/chatbot/letMeNERSCthatForYou/scripts/api_worker.sh

# Checks every 5 minutes if our worker is alive, if not restart it.
#SCRON --job-name=chatbot_worker_starter
#SCRON --account=nstaff
#SCRON --qos=cron
#SCRON --time=00:10:00
#SCRON --dependency=singleton
#SCRON -o /global/cfs/cdirs/nstaff/chatbot/letMeNERSCthatForYou/data/logs/api_worker/restarter_output-%j.out
#SCRON --open-mode=append
*/5 * * * * /global/cfs/cdirs/nstaff/chatbot/letMeNERSCthatForYou/scripts/api_worker_restarter.sh
```

## Various Scripts

* `local_chatbot.sh` starts a chatbot running on the current node,
* `api_chatbot.sh` starts a chatbot client calling the API for answers,
* `dev_local_chatbot.sh` starts the dev chatbot script running on the current node,
* `dev_check_retrieval.sh` runs a few closest-chunks queries and display the result, to test our retrieval pipeline.