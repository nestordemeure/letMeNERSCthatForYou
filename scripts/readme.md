# Additional Scripts

## SCRON Scripts

* `update_database.sh` when run, this scripts causes the updating of the database (git pull of the repo then refreshing of the vector database)
* `start_api_worker.sh` when run this script starts an API worker

The current SCRON file looks like this:

```shell
#SCRON -o /global/u2/n/nestor/scratch_perlmutter/chatbot/logs/output-%j.out                                                                                            
#SCRON --open-mode=append
0 3 * * * /global/u2/n/nestor/scratch_perlmutter/chatbot/update_database.sh
```

## Various Scripts

* `analysis.py` this script is used to get the approximate size of questions and answers (in tokens) for a given model