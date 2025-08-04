#!/bin/bash

# Auto-retry Mistral deployment script
# Runs every hour until successful, then notifies you

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_SCRIPT="$SCRIPT_DIR/deploy-mistral.sh"
LOG_FILE="$SCRIPT_DIR/auto-deploy.log"
STATUS_FILE="$SCRIPT_DIR/.deployment-status"
MAX_ATTEMPTS=168  # 7 days worth of hourly attempts

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}" | tee -a "$LOG_FILE"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}" | tee -a "$LOG_FILE"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}" | tee -a "$LOG_FILE"
}

# Send notification function
send_notification() {
    local status="$1"
    local message="$2"
    local instance_id="$3"
    local public_ip="$4"
    
    # Multiple notification methods
    
    # Method 1: Email via AWS SES (if configured)
    if command -v aws &> /dev/null; then
        aws ses send-email \
            --source "admin@asoba.co" \
            --destination ToAddresses="admin@asoba.co" \
            --message "Subject={Data='Mistral Deployment $status'},Body={Text={Data='$message\n\nInstance ID: $instance_id\nPublic IP: $public_ip\n\nTime: $(date)'}}" \
            2>/dev/null || log "Email notification failed (SES not configured)"
    fi
    
    # Method 2: Slack webhook (if you have one)
    SLACK_WEBHOOK="${SLACK_WEBHOOK_URL:-}"
    if [ -n "$SLACK_WEBHOOK" ]; then
        curl -X POST -H 'Content-type: application/json' \
            --data "{\"text\":\"🤖 Mistral Deployment $status\\n$message\\nInstance: $instance_id\\nIP: $public_ip\"}" \
            "$SLACK_WEBHOOK" 2>/dev/null || log "Slack notification failed"
    fi
    
    # Method 3: System notification (local desktop)
    if command -v notify-send &> /dev/null; then
        notify-send "Mistral Deployment $status" "$message"
    fi
    
    # Method 4: Write to a status file you can monitor
    echo "STATUS: $status" > "$STATUS_FILE"
    echo "MESSAGE: $message" >> "$STATUS_FILE"
    echo "INSTANCE_ID: $instance_id" >> "$STATUS_FILE"
    echo "PUBLIC_IP: $public_ip" >> "$STATUS_FILE"
    echo "TIMESTAMP: $(date)" >> "$STATUS_FILE"
    
    log "Notification sent: $status - $message"
}

# Check if already deployed
check_existing_deployment() {
    if [ -f ".mistral-base-instance-id" ]; then
        local existing_id=$(cat .mistral-base-instance-id)
        # Check both regions
        for region in us-east-1 us-west-2; do
            local state=$(aws ec2 describe-instances --region "$region" --instance-ids "$existing_id" --query 'Reservations[0].Instances[0].State.Name' --output text 2>/dev/null || echo "None")
            
            if [ "$state" = "running" ]; then
                local public_ip=$(aws ec2 describe-instances --region "$region" --instance-ids "$existing_id" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text 2>/dev/null || echo "None")
                log "Found existing running instance in $region: $existing_id (IP: $public_ip)"
                send_notification "ALREADY RUNNING" "Mistral base model instance was already deployed and running" "$existing_id" "$public_ip"
                exit 0
            fi
        done
    fi
}

# Single deployment attempt
attempt_deployment() {
    local attempt_num="$1"
    
    log "=== DEPLOYMENT ATTEMPT #$attempt_num ==="
    log "Starting deployment attempt at $(date)"
    
    # Run the deployment script and capture output
    if timeout 600 "$DEPLOY_SCRIPT" &>> "$LOG_FILE"; then
        log "✅ DEPLOYMENT SUCCESSFUL!"
        
        # Get instance details
        if [ -f ".mistral-base-instance-id" ]; then
            local instance_id=$(cat .mistral-base-instance-id)
            # Find which region the instance is in
            local region=""
            for r in us-east-1 us-west-2; do
                if aws ec2 describe-instances --region "$r" --instance-ids "$instance_id" &>/dev/null; then
                    region="$r"
                    break
                fi
            done
            
            local instance_info=$(aws ec2 describe-instances --region "$region" --instance-ids "$instance_id" --query 'Reservations[0].Instances[0].[PublicIpAddress,InstanceType,State.Name]' --output text 2>/dev/null || echo "None None None")
            local public_ip=$(echo "$instance_info" | cut -f1)
            local instance_type=$(echo "$instance_info" | cut -f2)
            local state=$(echo "$instance_info" | cut -f3)
            
            local success_message="Mistral base model deployment successful after $attempt_num attempts!
Instance Type: $instance_type
State: $state
Region: $region
Web Interface: http://$public_ip
API Endpoint: http://$public_ip/api/v1/completions
SSH: ssh -i ~/.ssh/mistral-base-key.pem ubuntu@$public_ip

The uncensored Mistral 7B v0.3 base model is now being set up. 
Initial setup takes 15-30 minutes. Check progress:
ssh -i ~/.ssh/mistral-base-key.pem ubuntu@$public_ip 'tail -f /var/log/user-data.log'"
            
            send_notification "SUCCESS" "$success_message" "$instance_id" "$public_ip"
            log "$success_message"
            return 0
        else
            warn "Deployment script succeeded but no instance ID found"
            return 1
        fi
    else
        warn "❌ Deployment attempt #$attempt_num failed"
        return 1
    fi
}

# Main retry loop
main() {
    log "Starting auto-deployment monitoring at $(date)"
    log "Will attempt deployment every hour for up to $MAX_ATTEMPTS attempts"
    log "Logs: $LOG_FILE"
    log "Status: $STATUS_FILE"
    
    # Check if already deployed
    check_existing_deployment
    
    local attempt=1
    
    while [ $attempt -le $MAX_ATTEMPTS ]; do
        log "--- Attempt $attempt of $MAX_ATTEMPTS ---"
        
        if attempt_deployment "$attempt"; then
            log "Deployment successful! Exiting auto-retry loop."
            exit 0
        fi
        
        if [ $attempt -lt $MAX_ATTEMPTS ]; then
            log "Attempt $attempt failed. Waiting 1 hour before next attempt..."
            log "Next attempt will be at $(date -d '+1 hour')"
            
            # Sleep for 1 hour (3600 seconds)
            sleep 3600
        fi
        
        ((attempt++))
    done
    
    # All attempts failed
    local failure_message="All $MAX_ATTEMPTS deployment attempts failed over 7 days. AWS GPU capacity may be severely constrained. Consider:
1. Requesting quota in different region (us-west-2)
2. Using CPU instance temporarily
3. Trying during off-peak hours manually"
    
    error "$failure_message"
    send_notification "FAILED" "$failure_message" "None" "None"
    exit 1
}

# Handle script interruption
cleanup() {
    log "Auto-deployment script interrupted. Status saved to $STATUS_FILE"
    exit 1
}

trap cleanup INT TERM

# Run main function
main "$@"