# Additional Scripts

## Scrontab Scripts

* `update_database.sh` causes the updating of the database (git pull of the repo then refreshing of the vector database),
* `start_api_worker.sh` starts an API worker.

The current [scrontab](https://docs.nersc.gov/jobs/workflow/scrontab/) file looks like this:

```shell
#SCRON --account=nstaff
#SCRON --time=00:10:00
#SCRON -o /global/cfs/cdirs/nstaff/chatbot/letMeNERSCthatForYou/data/logs/update_database/output-%j.out                                                                >
#SCRON --open-mode=append
0 3 * * * /global/cfs/cdirs/nstaff/chatbot/letMeNERSCthatForYou/scripts/update_database.sh
```

## Various Scripts

* `local_chatbot.sh` starts a chatbot running on the current node,
* `local_chatbot_dev.sh` starts the dev chatbot script running on the current node,
* `api_chatbot.sh` starts a chatbot calling the API for answers.
