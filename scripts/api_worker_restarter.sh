#!/bin/bash
# If our worker (`singleton_chatbot_worker`) is not running, then restart the script:
# /global/cfs/cdirs/nstaff/chatbot/letMeNERSCthatForYou/scripts/api_worker.sh

BUFFER_MIN=30 # Number of minutes to buffer before maintenance
SCRONTAB_NAME=singleton_chatbot_worker # Name of the scrontab job to check

MY_JOB=$(squeue --noheader -n $SCRONTAB_NAME --state=running,pending -u $USER -o "%12i %2t %9u %25j %6D %10M %12q %8f %18R")
IS_MY_JOB_RUNNING=$(echo "${MY_JOB}" | wc -c)

if [[ "$IS_MY_JOB_RUNNING" -ge 2 ]]; then
    printf "[%s] Job %s already in the queue \n" $(date "+%m-%d-%Y-%H:%M:%S") "$SCRONTAB_NAME"
    printf "===============================\n%s\n===============================\n" "$MY_JOB"
    exit 0;
fi

STR_TIME=$(scontrol show res -o | egrep 'login|workflow' | awk -F'[= ]' '{print $4}'| head -n 1)

if [[ "$STR_TIME" == "" ]]; then
    printf '[%s] No maintenance found\n' $(date "+%m-%d-%Y-%H:%M:%S")
    TIME_MIN_STR=""
else
    BAD_TIME=$(date "+%s" -d $STR_TIME)
    CUR_TIME=$(date "+%s")
    TIME_LEFT_MIN=$(((BAD_TIME-CUR_TIME)/60))

    printf '[%s] Minutes before next maintenance: %s\n' $(date "+%m-%d-%Y-%H:%M:%S") "$TIME_LEFT_MIN"

    TIME_MIN=$((TIME_LEFT_MIN-BUFFER_MIN))
    if [[ "$TIME_MIN" -le 0 ]]; then
        TIME_MIN_STR=""
    else
        TIME_MIN_STR="--time-min=${TIME_MIN}"
    fi
fi

# Run instead of printing for real script
printf '[%s] sbatch --dependency=singleton %s -q cron /global/cfs/cdirs/nstaff/chatbot/letMeNERSCthatForYou/scripts/api_worker.sh \n' $(date "+%m-%d-%Y-%H:%M:%S") "$TIME_MIN_STR"

unset ${!SLURM_@};
sbatch --dependency=singleton ${TIME_MIN_STR} -q cron /global/cfs/cdirs/nstaff/chatbot/letMeNERSCthatForYou/scripts/api_worker.sh