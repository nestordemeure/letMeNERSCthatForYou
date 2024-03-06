#!/bin/bash
# Purpose: Restart a specific worker script if it's not currently running and no restart job is pending or running,

# Configuration variables
JOB_NAME=singleton_chatbot_worker # Name of the original job to check in the SLURM queue
RESTART_JOB_NAME="${JOB_NAME}_restarted" # Name for restarted job
RESTART_JOB_INDICATOR="restart" # Indicator to identify restart jobs, can be part of the job name or description
END_TIME=$(date -d "tomorrow 01:15 AM PST" "+%s") # Target end time for the script
OUTPUT_PATH="/global/cfs/cdirs/nstaff/chatbot/letMeNERSCthatForYou/data/logs/api_worker/restarted_worker_output-%j.out" # Path for SLURM job output files

# Check if the original job is already running
ORIGINAL_JOB_CHECK=$(squeue --noheader -n $JOB_NAME --state=running -u $USER -o "%12i %2t %9u %25j %6D %10M %12q %8f %18R")
IS_ORIGINAL_JOB_RUNNING=$(echo "$ORIGINAL_JOB_CHECK" | grep -c '.')

# If the original job is running, log this and exit
if [[ "$IS_ORIGINAL_JOB_RUNNING" -ge 1 ]]; then
    printf "[%s] Original job %s is already running:\n" $(date "+%m-%d-%Y-%H:%M:%S") "$JOB_NAME"
    printf "%s\n" "$ORIGINAL_JOB_CHECK"
    printf "===============================\n"
    exit 0;
else
    printf "[%s] NO original job %s is running:\n" $(date "+%m-%d-%Y-%H:%M:%S") "$JOB_NAME"
    printf "> check:'%s' test:'%s'\n" "$ORIGINAL_JOB_CHECK" "$IS_ORIGINAL_JOB_RUNNING"
    printf "===============================\n"
fi

# Check if the restart job is already running or pending
RESTART_JOB_CHECK=$(squeue --noheader -n $RESTART_JOB_NAME --state=running,pending -u $USER -o "%12i %2t %9u %25j %6D %10M %12q %8f %18R")
IS_RESTART_JOB_RUNNING_OR_PENDING=$(echo "$RESTART_JOB_CHECK" | grep -c '.')

# If a restart job is running or already in the queue, log this and exit
if [[ "$IS_RESTART_JOB_RUNNING_OR_PENDING" -ge 1 ]]; then
    printf "[%s] Restart job %s is already running or in the queue:\n" $(date "+%m-%d-%Y-%H:%M:%S") "$RESTART_JOB_NAME"
    printf "%s\n" "$RESTART_JOB_CHECK"
    printf "===============================\n"
    exit 0;
else
    printf "[%s] NO restart job %s is running or pending:\n" $(date "+%m-%d-%Y-%H:%M:%S") "$RESTART_JOB_NAME"
    printf "> check:'%s' test:'%s'\n" "$RESTART_JOB_CHECK" "$IS_RESTART_JOB_RUNNING_OR_PENDING"
    printf "===============================\n"
fi

# Calculate the duration until the target end time
CURRENT_TIME=$(date "+%s")
DURATION=$((END_TIME - CURRENT_TIME))
DURATION_HMS=$(printf '%02d:%02d:%02d' $(($DURATION / 3600)) $(($DURATION % 3600 / 60)) $(($DURATION % 60)))

# Update the SBATCH command to include the new job name for restarts
SBATCH_COMMAND="sbatch --dependency=singleton --time=$DURATION_HMS --qos cron --output=$OUTPUT_PATH -J $RESTART_JOB_NAME /global/cfs/cdirs/nstaff/chatbot/letMeNERSCthatForYou/scripts/api_worker.sh"

# Echo the sbatch command for logging
printf '[%s] %s \n' $(date "+%m-%d-%Y-%H:%M:%S") "$SBATCH_COMMAND"

# Clear SLURM-related environment variables to prevent interference
unset ${!SLURM_@};

# Submit the job with sbatch
eval $SBATCH_COMMAND
printf "===============================\n"
