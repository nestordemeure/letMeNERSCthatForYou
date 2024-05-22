#!/bin/bash
# Purpose: Start the worker script if it's not currently running and no restart job is pending or running,

# Configuration variables
JOB_NAME=singleton_chatbot_worker # Name of the job to check in the SLURM queue
END_TIME=$(date -d 'tomorrow 01:15 AM' "+%s") # Target end time for the script
OUTPUT_PATH="/global/cfs/cdirs/nstaff/chatbot/letMeNERSCthatForYou/data/logs/api_worker/worker_output-%j.out" # Path for SLURM job output files

# Check if the job is already running or pending
JOB_CHECK=$(squeue --noheader -n $JOB_NAME --state=running,pending -u $USER -o "%12i %2t %9u %25j %6D %10M %12q %8f %18R")
IS_JOB_RUNNING_OR_PENDING=$(echo "$JOB_CHECK" | grep -c '.')

# If a restart job is running or already in the queue, log this and exit
if [[ "$IS_JOB_RUNNING_OR_PENDING" -ge 1 ]]; then
    printf "[%s] Job %s is already running or in the queue:\n" $(date "+%m-%d-%Y-%H:%M:%S") "$JOB_NAME"
    printf "%s\n" "$JOB_CHECK"
    printf "===============================\n"
    exit 0;
else
    printf "[%s] NO rob %s running or pending:\n" $(date "+%m-%d-%Y-%H:%M:%S") "$JOB_NAME"
    printf "> check:'%s' test:'%s'\n" "$JOB_CHECK" "$IS_JOB_RUNNING_OR_PENDING"
    printf "===============================\n"
fi

# Calculate the duration until the target end time
CURRENT_TIME=$(date "+%s")
DURATION=$((END_TIME - CURRENT_TIME))
DURATION_HMS=$(printf '%02d:%02d:%02d' $(($DURATION / 3600)) $(($DURATION % 3600 / 60)) $(($DURATION % 60)))

# Update the SBATCH command to include the new job name for restarts
SBATCH_COMMAND="sbatch --dependency=singleton --time=$DURATION_HMS --qos cron --output=$OUTPUT_PATH -J $JOB_NAME /global/cfs/cdirs/nstaff/chatbot/letMeNERSCthatForYou/scripts/api_worker.sh"

# Echo the sbatch command for logging
printf '[%s] %s \n' $(date "+%m-%d-%Y-%H:%M:%S") "$SBATCH_COMMAND"

# Clear SLURM-related environment variables to prevent interference
unset ${!SLURM_@};

# Submit the job with sbatch
eval $SBATCH_COMMAND
printf "===============================\n"
