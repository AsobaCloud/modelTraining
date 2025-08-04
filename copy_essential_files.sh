#!/bin/bash
set -euo pipefail

# Script to copy all essential corpus collection + training files
# From /home/shingai/sort/deployments to /home/shingai/sort/llm-training

SOURCE_DIR="/home/shingai/sort/deployments"
TARGET_DIR="/home/shingai/sort/llm-training"

echo "Copying essential files for corpus collection + training pipeline..."

# Create all necessary directories
mkdir -p "$TARGET_DIR"/{scripts/{corpus-generation/{iac-devops-corpus/{architecture-collectors,cicd-collectors,code-generation-collectors,corpus-builders,github-collectors,infrastructure-collectors,mcp-integration,validation},nsfw-corpus/{adult-scrapers,validation},policy-analyst-corpus/{academic-scrapers,insurance-scrapers,news-scrapers,policy-scrapers,validation},security-compliance-corpus/smart-collectors,utilities/{generic-scrapers,monitoring,validation}},mistral/{shared},qwen,monitoring},data/{collectors,corpus,validation},infrastructure,training,tests,config,docs}

echo "Created directory structure"

# Corpus Collection - IAC/DevOps (current versions)
cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/architecture-collectors/mermaid_real_collector.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/architecture-collectors/"
cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/architecture-collectors/smart_system_architecture_collector.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/architecture-collectors/"

cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/cicd-collectors/cicd_real_collector.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/cicd-collectors/"
cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/cicd-collectors/jupyter_real_collector.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/cicd-collectors/"
cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/cicd-collectors/smart_devops_automation_collector.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/cicd-collectors/"

cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/code-generation-collectors/data_engineering_collector.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/code-generation-collectors/"
cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/code-generation-collectors/smart_ai_ml_engineering_collector_v2.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/code-generation-collectors/"
cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/code-generation-collectors/smart_ai_training_inference_collector.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/code-generation-collectors/"
cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/code-generation-collectors/smart_code_generation_collector.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/code-generation-collectors/"
cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/code-generation-collectors/smart_collector_base.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/code-generation-collectors/"
cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/code-generation-collectors/smart_data_engineering_collector.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/code-generation-collectors/"
cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/code-generation-collectors/smart_technical_documentation_collector.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/code-generation-collectors/"

cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/corpus-builders/combine_iac_corpus.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/corpus-builders/"
cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/corpus-builders/create_final_iac_corpus.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/corpus-builders/"
cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/corpus-builders/integrate_enhanced_corpus.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/corpus-builders/"
cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/corpus-builders/mass_generator.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/corpus-builders/"

cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/github-collectors/comprehensive_repo_collector.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/github-collectors/"
cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/github-collectors/diverse_real_collector.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/github-collectors/"
cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/github-collectors/github_search_collector.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/github-collectors/"
cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/github-collectors/opensource_collector.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/github-collectors/"
cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/github-collectors/real_world_collector.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/github-collectors/"

cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/infrastructure-collectors/aws_cli_real_collector.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/infrastructure-collectors/"
cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/infrastructure-collectors/docker_real_collector.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/infrastructure-collectors/"
cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/infrastructure-collectors/helm_real_collector.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/infrastructure-collectors/"
cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/infrastructure-collectors/real_cdk_collector.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/infrastructure-collectors/"
cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/infrastructure-collectors/terraform_real_collector.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/infrastructure-collectors/"

cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/mcp-integration/iac_mcp_tools.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/mcp-integration/"
cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/mcp-integration/smart_asobacode_mcp_collector.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/mcp-integration/"
cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/mcp-integration/targeted_collector.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/mcp-integration/"

cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/validation/test_cicd_collector.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/validation/"
cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/validation/test_helm_collector.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/validation/"
cp "$SOURCE_DIR/scripts/corpus-generation/iac-devops-corpus/validation/test_mermaid_collector.py" "$TARGET_DIR/scripts/corpus-generation/iac-devops-corpus/validation/"

# Corpus Collection - NSFW
cp "$SOURCE_DIR/scripts/corpus-generation/nsfw-corpus/adult-scrapers/babesource_scraper.py" "$TARGET_DIR/scripts/corpus-generation/nsfw-corpus/adult-scrapers/"
cp "$SOURCE_DIR/scripts/corpus-generation/nsfw-corpus/adult-scrapers/imagefap_scraper.py" "$TARGET_DIR/scripts/corpus-generation/nsfw-corpus/adult-scrapers/"
cp "$SOURCE_DIR/scripts/corpus-generation/nsfw-corpus/validation/test_babesource_scraper.py" "$TARGET_DIR/scripts/corpus-generation/nsfw-corpus/validation/"
cp "$SOURCE_DIR/scripts/corpus-generation/nsfw-corpus/NSFW_CORPUS_DOCUMENTATION.md" "$TARGET_DIR/scripts/corpus-generation/nsfw-corpus/"

# Corpus Collection - Policy Analyst (current versions)
cp "$SOURCE_DIR/scripts/corpus-generation/policy-analyst-corpus/academic-scrapers/scihub_enhanced_scraper.py" "$TARGET_DIR/scripts/corpus-generation/policy-analyst-corpus/academic-scrapers/"
cp "$SOURCE_DIR/scripts/corpus-generation/policy-analyst-corpus/insurance-scrapers/industrial_insurance_scraper.py" "$TARGET_DIR/scripts/corpus-generation/policy-analyst-corpus/insurance-scrapers/"
cp "$SOURCE_DIR/scripts/corpus-generation/policy-analyst-corpus/insurance-scrapers/expand_insurance_corpus.py" "$TARGET_DIR/scripts/corpus-generation/policy-analyst-corpus/insurance-scrapers/"
cp "$SOURCE_DIR/scripts/corpus-generation/policy-analyst-corpus/insurance-scrapers/run_insurance_collection.py" "$TARGET_DIR/scripts/corpus-generation/policy-analyst-corpus/insurance-scrapers/"
cp "$SOURCE_DIR/scripts/corpus-generation/policy-analyst-corpus/news-scrapers/news_2025_scraper.py" "$TARGET_DIR/scripts/corpus-generation/policy-analyst-corpus/news-scrapers/"
cp "$SOURCE_DIR/scripts/corpus-generation/policy-analyst-corpus/policy-scrapers/government_officials_scraper.py" "$TARGET_DIR/scripts/corpus-generation/policy-analyst-corpus/policy-scrapers/"
cp "$SOURCE_DIR/scripts/corpus-generation/policy-analyst-corpus/policy-scrapers/policy_2025_scraper.py" "$TARGET_DIR/scripts/corpus-generation/policy-analyst-corpus/policy-scrapers/"
cp "$SOURCE_DIR/scripts/corpus-generation/policy-analyst-corpus/validation/test_insurance_scraper.py" "$TARGET_DIR/scripts/corpus-generation/policy-analyst-corpus/validation/"

# Corpus Collection - Security/Compliance (current version)
cp "$SOURCE_DIR/scripts/corpus-generation/security-compliance-corpus/smart-collectors/smart_security_compliance_collector_v2.py" "$TARGET_DIR/scripts/corpus-generation/security-compliance-corpus/smart-collectors/"

# Corpus Collection - Utilities
cp "$SOURCE_DIR/scripts/corpus-generation/utilities/generic-scrapers/add_metadata_labels.py" "$TARGET_DIR/scripts/corpus-generation/utilities/generic-scrapers/"
cp "$SOURCE_DIR/scripts/corpus-generation/utilities/generic-scrapers/improved_scraper.py" "$TARGET_DIR/scripts/corpus-generation/utilities/generic-scrapers/"
cp "$SOURCE_DIR/scripts/corpus-generation/utilities/generic-scrapers/multi_site_collector.py" "$TARGET_DIR/scripts/corpus-generation/utilities/generic-scrapers/"
cp "$SOURCE_DIR/scripts/corpus-generation/utilities/monitoring/monitor_smart_collectors.py" "$TARGET_DIR/scripts/corpus-generation/utilities/monitoring/"
cp "$SOURCE_DIR/scripts/corpus-generation/utilities/validation/test_improved_scraper.py" "$TARGET_DIR/scripts/corpus-generation/utilities/validation/"
cp "$SOURCE_DIR/scripts/corpus-generation/README.md" "$TARGET_DIR/scripts/corpus-generation/"
cp "$SOURCE_DIR/scripts/corpus-generation/SCRAPER_SUITE_SUMMARY.md" "$TARGET_DIR/scripts/corpus-generation/"

# Data Processing & Assembly
cp "$SOURCE_DIR/data/collectors/advanced_corpus_builder.py" "$TARGET_DIR/data/collectors/"
cp "$SOURCE_DIR/data/collectors/analyze_scraping_strategy.py" "$TARGET_DIR/data/collectors/"
cp "$SOURCE_DIR/data/collectors/aws_cli_collector.py" "$TARGET_DIR/data/collectors/"
cp "$SOURCE_DIR/data/collectors/collect_all_datasets.sh" "$TARGET_DIR/data/collectors/"
cp "$SOURCE_DIR/data/collectors/download_figshare_dataset.py" "$TARGET_DIR/data/collectors/"
cp "$SOURCE_DIR/data/collectors/evidence_first_corpus_enhancer.py" "$TARGET_DIR/data/collectors/"
cp "$SOURCE_DIR/data/collectors/evidence_or_silence_examples.py" "$TARGET_DIR/data/collectors/"
cp "$SOURCE_DIR/data/collectors/generate_policy_corpus.py" "$TARGET_DIR/data/collectors/"
cp "$SOURCE_DIR/data/collectors/policy_pdf_processor.py" "$TARGET_DIR/data/collectors/"
cp "$SOURCE_DIR/data/collectors/quick_shell_corpus.py" "$TARGET_DIR/data/collectors/"
cp "$SOURCE_DIR/data/collectors/run_full_babesource_scraping.py" "$TARGET_DIR/data/collectors/"
cp "$SOURCE_DIR/data/collectors/shell_corpus_builder.py" "$TARGET_DIR/data/collectors/"
cp "$SOURCE_DIR/data/collectors/shell_corpus_summary.md" "$TARGET_DIR/data/collectors/"
cp "$SOURCE_DIR/data/collectors/test_babesource_small.py" "$TARGET_DIR/data/collectors/"
cp "$SOURCE_DIR/data/validation/universal_validation_pipeline.py" "$TARGET_DIR/data/validation/"

# Existing Corpus Data
cp "$SOURCE_DIR/data/corpus"/*.jsonl "$TARGET_DIR/data/corpus/" 2>/dev/null || echo "Some corpus files may not exist, continuing..."

# Mistral Training Pipeline
cp "$SOURCE_DIR/scripts/mistral/prepare_mistral_dataset.py" "$TARGET_DIR/scripts/mistral/"
cp "$SOURCE_DIR/scripts/mistral/train_mistral_simple.py" "$TARGET_DIR/scripts/mistral/"
cp "$SOURCE_DIR/scripts/mistral/shared/dataset_utils.py" "$TARGET_DIR/scripts/mistral/shared/"
cp "$SOURCE_DIR/scripts/mistral/shared/__init__.py" "$TARGET_DIR/scripts/mistral/shared/"
cp "$SOURCE_DIR/scripts/mistral/__init__.py" "$TARGET_DIR/scripts/mistral/"
cp "$SOURCE_DIR/scripts/mistral/deploy_mistral_to_g5.sh" "$TARGET_DIR/scripts/mistral/"
cp "$SOURCE_DIR/scripts/mistral/bootstrap_mistral_deployment.sh" "$TARGET_DIR/scripts/mistral/"
cp "$SOURCE_DIR/scripts/mistral/mistral_ami_user_data.sh" "$TARGET_DIR/scripts/mistral/"
cp "$SOURCE_DIR/scripts/mistral/MISTRAL_TRAINING_GUIDE.md" "$TARGET_DIR/scripts/mistral/"
cp "$SOURCE_DIR/scripts/mistral/MISTRAL_POLICY_CORPUS_COMPLETION_REPORT.md" "$TARGET_DIR/scripts/mistral/"
cp "$SOURCE_DIR/scripts/mistral/complete_mistral_qlora_training_guide.md" "$TARGET_DIR/scripts/mistral/"
cp "$SOURCE_DIR/scripts/mistral/mistral_golden_config_documentation.md" "$TARGET_DIR/scripts/mistral/"
cp "$SOURCE_DIR/scripts/mistral/validate_mistral_golden_config.py" "$TARGET_DIR/scripts/mistral/"

# Qwen Training Pipeline (current versions)
cp "$SOURCE_DIR/scripts/qwen/train_qwen_golden_config.py" "$TARGET_DIR/scripts/qwen/"
cp "$SOURCE_DIR/scripts/qwen/deploy_qwen_verbosity_training_to_gpu.sh" "$TARGET_DIR/scripts/qwen/"
cp "$SOURCE_DIR/scripts/qwen/bootstrap_qwen_claude_deployment.sh" "$TARGET_DIR/scripts/qwen/"
cp "$SOURCE_DIR/scripts/qwen/run_unsloth_training_monitored_fixed.sh" "$TARGET_DIR/scripts/qwen/"
cp "$SOURCE_DIR/scripts/qwen/qwen_config.json" "$TARGET_DIR/scripts/qwen/"
cp "$SOURCE_DIR/scripts/qwen/qwen_inference_server.py" "$TARGET_DIR/scripts/qwen/"
cp "$SOURCE_DIR/scripts/qwen/qwen_inference_server_awq.py" "$TARGET_DIR/scripts/qwen/"
cp "$SOURCE_DIR/scripts/qwen/qwen_inference_server_lora.py" "$TARGET_DIR/scripts/qwen/"
cp "$SOURCE_DIR/scripts/qwen/test_qwen_claude_md.py" "$TARGET_DIR/scripts/qwen/"
cp "$SOURCE_DIR/scripts/qwen/test_qwen_prompt_completion.py" "$TARGET_DIR/scripts/qwen/"
cp "$SOURCE_DIR/scripts/qwen/test_data_compatibility.py" "$TARGET_DIR/scripts/qwen/"
cp "$SOURCE_DIR/scripts/qwen/validate_qwen_styles.py" "$TARGET_DIR/scripts/qwen/"
cp "$SOURCE_DIR/scripts/qwen/QWEN_TRAINING_GUIDE.md" "$TARGET_DIR/scripts/qwen/"
cp "$SOURCE_DIR/scripts/qwen/QWEN_CORPUS_COMPLETION_REPORT.md" "$TARGET_DIR/scripts/qwen/"
cp "$SOURCE_DIR/scripts/qwen/qwen_claude_md_system_prompt_concise.txt" "$TARGET_DIR/scripts/qwen/"
cp "$SOURCE_DIR/scripts/qwen/iac_claude_md_system_prompt.txt" "$TARGET_DIR/scripts/qwen/"
cp "$SOURCE_DIR/scripts/qwen/qwen_iac_fine_tuning_plan_corrected.md" "$TARGET_DIR/scripts/qwen/"
cp "$SOURCE_DIR/scripts/qwen/qwen_training_domain_expansion.md" "$TARGET_DIR/scripts/qwen/"

# Monitoring & Validation
cp "$SOURCE_DIR/scripts/monitoring/monitor_training.py" "$TARGET_DIR/scripts/monitoring/"
cp "$SOURCE_DIR/scripts/monitoring/training_monitor.py" "$TARGET_DIR/scripts/monitoring/"
cp "$SOURCE_DIR/scripts/monitoring/s3_model_uploader.py" "$TARGET_DIR/scripts/monitoring/"
cp "$SOURCE_DIR/scripts/monitoring/test_monitor_integration.py" "$TARGET_DIR/scripts/monitoring/"
cp "$SOURCE_DIR/scripts/monitoring/validate_deployment.py" "$TARGET_DIR/scripts/monitoring/"

# Infrastructure & Deployment
cp "$SOURCE_DIR/infrastructure/deploy-mistral-us-west-2.sh" "$TARGET_DIR/infrastructure/"
cp "$SOURCE_DIR/infrastructure/deploy-policy-training-instance.sh" "$TARGET_DIR/infrastructure/"
cp "$SOURCE_DIR/infrastructure/setup_qlora_instance.sh" "$TARGET_DIR/infrastructure/"
cp "$SOURCE_DIR/infrastructure/auto-deploy-mistral.sh" "$TARGET_DIR/infrastructure/"
cp "$SOURCE_DIR/infrastructure/flux_one_shot.sh" "$TARGET_DIR/infrastructure/"

# S3 Model Streaming
cp "$SOURCE_DIR/scripts/stream_mistral_small_24b_to_s3.py" "$TARGET_DIR/scripts/"
cp "$SOURCE_DIR/scripts/stream_mistral_small_24b_background.sh" "$TARGET_DIR/scripts/"
cp "$SOURCE_DIR/scripts/stream_qwen3_coder_30b_to_s3.py" "$TARGET_DIR/scripts/"
cp "$SOURCE_DIR/scripts/stream_qwen3_coder_30b_background.sh" "$TARGET_DIR/scripts/"

# Tests
cp "$SOURCE_DIR/tests/test_data_pipeline.py" "$TARGET_DIR/tests/"
cp "$SOURCE_DIR/tests/test_training_script.py" "$TARGET_DIR/tests/"
cp "$SOURCE_DIR/tests/test_integration.py" "$TARGET_DIR/tests/"
cp "$SOURCE_DIR/tests/test_infrastructure.py" "$TARGET_DIR/tests/"
cp "$SOURCE_DIR/tests/test_training_monitor.py" "$TARGET_DIR/tests/"
cp "$SOURCE_DIR/tests/__init__.py" "$TARGET_DIR/tests/"

# Training Framework
cp "$SOURCE_DIR/training/qlora_trainer.py" "$TARGET_DIR/training/"
cp "$SOURCE_DIR/training/policy_analysis_qlora_trainer.py" "$TARGET_DIR/training/"
cp "$SOURCE_DIR/training/iac_qlora_trainer.py" "$TARGET_DIR/training/"

# Configuration
cp "$SOURCE_DIR/config/qlora_config.json" "$TARGET_DIR/config/"
cp "$SOURCE_DIR/config/flux_training_config.json" "$TARGET_DIR/config/"
cp "$SOURCE_DIR/config/iac_qlora_config.json" "$TARGET_DIR/config/"
cp "$SOURCE_DIR/config/validated_flux_config.json" "$TARGET_DIR/config/"

# Current & Relevant Documentation
cp "$SOURCE_DIR/CLAUDE.md" "$TARGET_DIR/"
cp "$SOURCE_DIR/docs/operatives_last_processing.md" "$TARGET_DIR/docs/" 2>/dev/null || echo "operatives_last_processing.md not found, skipping"

# Package Structure
cp "$SOURCE_DIR/scripts/__init__.py" "$TARGET_DIR/scripts/"

# Create .gitignore
cat > "$TARGET_DIR/.gitignore" << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual environments
venv/
env/
ENV/
env.bak/
venv.bak/
scripts/mistral/venv/
scripts/qwen/venv/

# Training artifacts
*.bin
*.safetensors
*.pt
*.pth
checkpoints/
wandb/
logs/
*.log

# Data
*.jsonl.bak
*.csv.bak
temp/
tmp/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# AWS
.aws/

# Environment variables
.env
EOF

echo "✅ Successfully copied all essential files for corpus collection + training pipeline!"
echo "📁 Repository created at: $TARGET_DIR"
echo "🔧 Next steps:"
echo "   1. cd $TARGET_DIR"
echo "   2. git init && git branch -m main"
echo "   3. git add ."
echo "   4. git commit -m 'Initial commit: LLM training pipeline'"