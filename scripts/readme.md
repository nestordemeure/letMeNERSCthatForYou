# Additional Scripts

## SCRON Scripts

* `local_chatbot.sh` starts a chatbot running on the current node,
* `update_database.sh` causes the updating of the database (git pull of the repo then refreshing of the vector database),
* `start_api_worker.sh` starts an API worker.

The current SCRON file looks like this:

```shell
#SCRON -o /global/cfs/cdirs/nstaff/chatbot/letMeNERSCthatForYou/data/logs/update_database/output-%j.out                                                                                            
#SCRON --open-mode=append
0 3 * * * /global/cfs/cdirs/nstaff/chatbot/letMeNERSCthatForYou/scripts/update_database.sh
```

## Various Scripts

* `analysis.py` this script is used to get the approximate size of questions and answers (in tokens) for a given model