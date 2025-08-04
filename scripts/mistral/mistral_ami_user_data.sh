#!/bin/bash
# EC2 User Data script for automatic Mistral deployment bootstrap
# Use this when launching instances from ami-0a39335458731538a (Mistral) or ami-0d23439fbd78468a2 (Flux)

# Download and execute bootstrap script
cd /home/ubuntu
aws s3 cp s3://asoba-llm-cache/scripts/bootstrap_mistral_deployment.sh . --region us-east-1
chmod +x bootstrap_mistral_deployment.sh
sudo ./bootstrap_mistral_deployment.sh 2>&1 | tee bootstrap.log

# Send completion notification (optional)
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
echo "Instance $INSTANCE_ID bootstrap completed at $(date)" >> /var/log/deployment-complete.log