#!/bin/bash
# Validated Pipeline Deployment
# Replaces trial-and-error with systematic validation

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
    exit 1
}

info() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

show_usage() {
    cat << EOF
Validated Pipeline Deployment

USAGE:
    $0 --instance-id <id> --instance-ip <ip> --ssh-key <path> [options]

REQUIRED:
    --instance-id       EC2 instance ID (e.g., i-01fa5b57d64c6196a)
    --instance-ip       Instance IP address (e.g., 54.197.142.172)  
    --ssh-key          Path to SSH key (e.g., config/mistral-base.pem)

OPTIONS:
    --skip-validation   Skip pre-flight validation (NOT RECOMMENDED)
    --force            Deploy even if validation fails (DANGEROUS)
    --validation-only   Only run validation, don't deploy
    --json-report      Save validation report to JSON file

EXAMPLES:
    # Full validated deployment (RECOMMENDED):
    $0 --instance-id i-01fa5b57d64c6196a --instance-ip 54.197.142.172 --ssh-key config/mistral-base.pem

    # Validation only:
    $0 --instance-id i-01fa5b57d64c6196a --instance-ip 54.197.142.172 --ssh-key config/mistral-base.pem --validation-only

    # Force deployment (DANGEROUS):
    $0 --instance-id i-01fa5b57d64c6196a --instance-ip 54.197.142.172 --ssh-key config/mistral-base.pem --force

VALIDATION CHECKS:
    ✅ Infrastructure (AWS, S3, Instance, SSH, Disk)
    ✅ Environment (Python, PyTorch, CUDA, GPU, Packages)  
    ✅ Data Pipeline (Sources, Format, Network)
    ✅ Training Setup (Memory, Model, Scripts, Checkpoints)
    ✅ Monitoring (Slack, S3, Heartbeat)

SUCCESS CRITERIA:
    • Readiness Score ≥ 90/100
    • Zero critical failures
    • All infrastructure checks pass
    • All environment checks pass

EOF
}

parse_args() {
    INSTANCE_ID=""
    INSTANCE_IP=""
    SSH_KEY=""
    SKIP_VALIDATION=false
    FORCE_DEPLOY=false
    VALIDATION_ONLY=false
    JSON_REPORT=""
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --instance-id)
                INSTANCE_ID="$2"
                shift 2
                ;;
            --instance-ip)
                INSTANCE_IP="$2"
                shift 2
                ;;
            --ssh-key)
                SSH_KEY="$2"
                shift 2
                ;;
            --skip-validation)
                SKIP_VALIDATION=true
                shift
                ;;
            --force)
                FORCE_DEPLOY=true
                shift
                ;;
            --validation-only)
                VALIDATION_ONLY=true
                shift
                ;;
            --json-report)
                JSON_REPORT="$2"
                shift 2
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                error "Unknown argument: $1"
                ;;
        esac
    done
    
    # Validate required arguments
    if [[ -z "$INSTANCE_ID" ]] || [[ -z "$INSTANCE_IP" ]] || [[ -z "$SSH_KEY" ]]; then
        error "Missing required arguments. Use --help for usage."
    fi
    
    if [[ ! -f "$SSH_KEY" ]]; then
        error "SSH key not found: $SSH_KEY"
    fi
}

run_pre_flight_validation() {
    log "🚀 Running Pre-Flight Validation"
    echo "============================================================"
    
    local json_arg=""
    if [[ -n "$JSON_REPORT" ]]; then
        json_arg="--json-output $JSON_REPORT"
    fi
    
    if python3 "$SCRIPT_DIR/preflight_check.py" \
        --instance-id "$INSTANCE_ID" \
        --instance-ip "$INSTANCE_IP" \
        --ssh-key "$SSH_KEY" \
        $json_arg; then
        
        log "✅ Pre-flight validation PASSED"
        return 0
    else
        error "❌ Pre-flight validation FAILED"
        return 1
    fi
}

run_pipeline_deployment() {
    log "🚀 Starting Pipeline Deployment"
    echo "============================================================"
    
    # Run the actual training pipeline
    if "$SCRIPT_DIR/../mistral/run_mistral_training_pipeline.sh" \
        --instance-id "$INSTANCE_ID" \
        --instance-ip "$INSTANCE_IP" \
        --ssh-key "$SSH_KEY"; then
        
        log "✅ Pipeline deployment COMPLETED"
        return 0
    else
        error "❌ Pipeline deployment FAILED"
        return 1
    fi
}

show_deployment_summary() {
    echo ""
    echo "============================================================"
    log "📊 DEPLOYMENT SUMMARY"
    echo "============================================================"
    
    if [[ -n "$JSON_REPORT" ]] && [[ -f "$JSON_REPORT" ]]; then
        # Parse JSON report for summary
        local readiness_score=$(python3 -c "import json; print(json.load(open('$JSON_REPORT'))['readiness_score'])")
        local deployment_ready=$(python3 -c "import json; print(json.load(open('$JSON_REPORT'))['deployment_ready'])")
        
        echo "Validation Report: $JSON_REPORT"
        echo "Readiness Score: $readiness_score/100"
        echo "Deployment Ready: $deployment_ready"
    fi
    
    echo "Instance: $INSTANCE_ID ($INSTANCE_IP)"
    echo "SSH Key: $SSH_KEY"
    echo "Completed: $(date)"
    echo ""
    
    if [[ "$VALIDATION_ONLY" == "true" ]]; then
        log "Validation complete. Use deployment command when ready:"
        echo "    $0 --instance-id $INSTANCE_ID --instance-ip $INSTANCE_IP --ssh-key $SSH_KEY"
    else
        log "Monitor training progress:"
        echo "    python3 scripts/monitoring/monitor.py --run-id \$(ls -t mistral_training_info.txt 2>/dev/null | head -1 | grep -o 'mistral-[0-9]*-[0-9]*' || echo 'unknown')"
        echo ""
        log "SSH to instance:"
        echo "    ssh -i $SSH_KEY ubuntu@$INSTANCE_IP"
    fi
}

main() {
    info "🧪 VALIDATED PIPELINE DEPLOYMENT SYSTEM"
    info "No more trial-and-error deployments!"
    echo ""
    
    parse_args "$@"
    
    # Phase 1: Pre-flight validation (unless skipped)
    if [[ "$SKIP_VALIDATION" == "false" ]]; then
        if ! run_pre_flight_validation; then
            if [[ "$FORCE_DEPLOY" == "true" ]]; then
                warn "⚠️  FORCING DEPLOYMENT despite validation failures!"
                warn "⚠️  This is DANGEROUS and may result in wasted time!"
            else
                error "🚨 DEPLOYMENT BLOCKED by validation failures"
                echo ""
                echo "Fix the issues above and try again, or use --force to override (not recommended)"
                exit 1
            fi
        fi
    else
        warn "⚠️  SKIPPING PRE-FLIGHT VALIDATION (not recommended)"
    fi
    
    # Phase 2: Deployment (unless validation-only)
    if [[ "$VALIDATION_ONLY" == "false" ]]; then
        log "🚀 Validation passed, proceeding with deployment..."
        echo ""
        
        if ! run_pipeline_deployment; then
            error "🚨 DEPLOYMENT FAILED"
            echo ""
            echo "Check the logs above for specific errors."
            echo "This failure was unexpected since validation passed."
            echo "Please report this as a test coverage gap."
            exit 1
        fi
    fi
    
    # Phase 3: Summary
    show_deployment_summary
    
    if [[ "$VALIDATION_ONLY" == "true" ]]; then
        log "🎉 VALIDATION COMPLETE"
    else
        log "🎉 DEPLOYMENT COMPLETE"
    fi
}

# Trap to ensure cleanup on exit
trap 'echo -e "\n${RED}Deployment interrupted${NC}"; exit 130' INT TERM

main "$@"