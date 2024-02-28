#!/bin/bash
# Purpose: Restart a specific worker script if it's not currently running,
# taking into account system maintenance schedules and ensuring output is logged appropriately.

# Configuration variables
BUFFER_BEFORE_MAINTENANCE_MIN=30 # Buffer time in minutes before system maintenance begins
JOB_NAME=singleton_chatbot_worker # Name of the job to check in the SLURM queue
RESTART_JOB_INDICATOR="restart" # Indicator to identify restart jobs, can be part of the job name or description
END_TIME=$(date -d "tomorrow 01:15 AM PST" "+%s") # Target end time for the script
OUTPUT_PATH="/global/cfs/cdirs/nstaff/chatbot/letMeNERSCthatForYou/data/logs/api_worker/restarted_worker_output-%j.out" # Path for SLURM job output files

# Check if the specified job is already running or pending in the queue
JOB_CHECK=$(squeue --noheader -n $JOB_NAME --state=running,pending -u $USER -o "%12i %2t %9u %25j %6D %10M %12q %8f %18R")
IS_JOB_RUNNING_OR_PENDING=$(echo "$JOB_CHECK" | wc -l)

# If a job is running or already in the queue, log this and exit
if [[ "$IS_JOB_RUNNING_OR_PENDING" -ge 1 ]]; then
    printf "[%s] Job %s is already running or in the queue\n" $(date "+%m-%d-%Y-%H:%M:%S") "$JOB_NAME"
    printf "===============================\n%s\n===============================\n" "$JOB_CHECK"
    exit 0;
fi

# Check for scheduled system maintenance
MAINTENANCE_TIME=$(scontrol show res -o | egrep 'login|workflow' | awk -F'[= ]' '{print $4}' | head -n 1)

# If no maintenance is scheduled, proceed without adjusting for maintenance
if [[ "$MAINTENANCE_TIME" == "" ]]; then
    printf '[%s] No maintenance found\n' $(date "+%m-%d-%Y-%H:%M:%S")
    TIME_MIN_STR=""
else
    # Calculate time until maintenance starts
    MAINTENANCE_START=$(date "+%s" -d $MAINTENANCE_TIME)
    CURRENT_TIME=$(date "+%s")
    TIME_UNTIL_MAINTENANCE_MIN=$(((MAINTENANCE_START-CURRENT_TIME)/60))

    printf '[%s] Minutes before next maintenance: %s\n' $(date "+%m-%d-%Y-%H:%M:%S") "$TIME_UNTIL_MAINTENANCE_MIN"

    # Adjust for buffer before maintenance
    TIME_MIN=$((TIME_UNTIL_MAINTENANCE_MIN-BUFFER_BEFORE_MAINTENANCE_MIN))
    if [[ "$TIME_MIN" -le 0 ]]; then
        TIME_MIN_STR=""
    else
        TIME_MIN_STR="--time-min=${TIME_MIN}"
    fi
fi

# Calculate the duration until the target end time
CURRENT_TIME=$(date "+%s")
DURATION=$((END_TIME - CURRENT_TIME))
DURATION_HMS=$(printf '%02d:%02d:%02d' $(($DURATION / 3600)) $(($DURATION % 3600 / 60)) $(($DURATION % 60)))

# Prepare the sbatch command with calculated constraints and output path, including a unique identifier for restart jobs
SBATCH_COMMAND="sbatch --dependency=singleton ${TIME_MIN_STR} --time=$DURATION_HMS -q shared --output=$OUTPUT_PATH -J $JOB_NAME /global/cfs/cdirs/nstaff/chatbot/letMeNERSCthatForYou/scripts/api_worker.sh"

# Echo the sbatch command for logging
printf '[%s] %s \n' $(date "+%m-%d-%Y-%H:%M:%S") "$SBATCH_COMMAND"

# Clear SLURM-related environment variables to prevent interference
unset ${!SLURM_@};

# Submit the job with sbatch
eval $SBATCH_COMMAND
