#!/bin/bash
set -euo pipefail

# Comprehensive NSFW Dataset Collection Script
# Collects from Figshare + multiple sites to reach 1000 images

BUCKET="flux-dev-nsfw"
REGION="us-east-1"
WORKERS=10

echo "=== NSFW Dataset Collection Pipeline ==="
echo "Target: 1000 images for Flux training"
echo "Bucket: s3://$BUCKET"
echo ""

# Check current dataset size
echo "Checking current dataset size..."
CURRENT_COUNT=$(aws s3 ls s3://$BUCKET/ --region $REGION --recursive | wc -l || echo "0")
echo "Current images: $CURRENT_COUNT"
echo ""

# Phase 1: Download Figshare dataset (likely largest source)
echo "=== Phase 1: Figshare Adult Content Dataset ==="
python3 scripts/download_figshare_dataset.py \
    --bucket "$BUCKET" \
    --region "$REGION" \
    --workers "$WORKERS" \
    --max-images 600

# Check progress
FIGSHARE_COUNT=$(aws s3 ls s3://$BUCKET/ --region $REGION --recursive | wc -l)
echo "After Figshare: $FIGSHARE_COUNT images"
echo ""

# Phase 2: Multi-site collection (if we need more)
if [ "$FIGSHARE_COUNT" -lt 800 ]; then
    echo "=== Phase 2: Multi-Site Collection ==="
    echo "Need more images, running multi-site collector..."
    
    python3 scripts/multi_site_collector.py \
        --bucket "$BUCKET" \
        --region "$REGION" \
        --images-per-url 30 \
        --workers 5
    
    FINAL_COUNT=$(aws s3 ls s3://$BUCKET/ --region $REGION --recursive | wc -l)
    echo "After multi-site: $FINAL_COUNT images"
else
    echo "Figshare dataset provided sufficient images, skipping multi-site collection"
    FINAL_COUNT=$FIGSHARE_COUNT
fi

echo ""
echo "=== Phase 3: Dataset Validation ==="
python3 scripts/validate_dataset.py \
    --bucket "$BUCKET" \
    --region "$REGION" \
    --sample-size 200

echo ""
echo "=== Collection Complete ==="
echo "Final dataset size: $FINAL_COUNT images"
echo "Target reached: $([ "$FINAL_COUNT" -ge 1000 ] && echo "YES" || echo "NO ($((1000 - FINAL_COUNT)) more needed)")"
echo ""
echo "Next steps:"
echo "1. Review validation report: aws s3 cp s3://$BUCKET/validation_report.json - --region $REGION | jq ."
echo "2. Start training preparation: follow docs/nsfw_flux_training_plan.md"
echo "3. Set up captioning with joy-caption-batch"