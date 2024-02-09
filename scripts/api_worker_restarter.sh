#!/bin/bash
# If our worker (`singleton_chatbot_worker`) is not running, then restart the script:
# /global/cfs/cdirs/nstaff/chatbot/letMeNERSCthatForYou/scripts/api_worker.sh

BUFFER_MIN=30 # Number of minutes to buffer before maintenance
SCRONTAB_NAME=singleton_chatbot_worker # Name of the scrontab job to check
END_TIME=$(date -d "tomorrow 01:15 AM PST" "+%s") # When should the script stop running

# NOTE: we don't use --state=running,pending as pending would detect tomorow's run in the queue
MY_JOB=$(squeue --noheader -n $SCRONTAB_NAME --state=running -u $USER -o "%12i %2t %9u %25j %6D %10M %12q %8f %18R")
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

# Calculate Duration until target end time
current_time=$(date "+%s")
duration=$((END_TIME - current_time))
duration_hms=$(printf '%02d:%02d:%02d' $(($duration / 3600)) $(($duration % 3600 / 60)) $(($duration % 60)))

# Run instead of printing for real script
printf '[%s] sbatch --dependency=singleton %s --time=%s -q cron /global/cfs/cdirs/nstaff/chatbot/letMeNERSCthatForYou/scripts/api_worker.sh \n' $(date "+%m-%d-%Y-%H:%M:%S") "$TIME_MIN_STR" "$duration_hms"

unset ${!SLURM_@};
sbatch --dependency=singleton ${TIME_MIN_STR} --time=$duration_hms -q cron /global/cfs/cdirs/nstaff/chatbot/letMeNERSCthatForYou/scripts/api_worker.sh
