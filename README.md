# Asoba Model Training Pipeline

**Production-ready tooling and utilities for custom LLM training, fine-tuning, and deployment.**

## Overview

This repository contains Asoba's complete infrastructure for training custom language models. Currently supports **Qwen** and **Mistral** model families, with extensible architecture for future model integrations.

### Core Capabilities

- **Data Collection & Corpus Building** - Automated scrapers and collectors for domain-specific training data
- **Training Configurations** - Hardware-optimized configs mapping models to tech stacks and instance types  
- **One-Shot Training Scripts** - Streamlined deployment across various AWS instance types
- **Monitoring & Validation** - Real-time training progress tracking and quality assurance

## Quick Start

### Current Model Support

| Model Family | Status | Hardware | Config |
|--------------|--------|----------|--------|
| **Qwen** | ✅ Production | g5.xlarge+ | [qwen/](./scripts/qwen/) |
| **Mistral** | ✅ Production | g5.2xlarge+ | [mistral/](./scripts/mistral/) |

### Training a Model

```bash
# Qwen training (recommended)
./scripts/qwen/deploy_qwen_verbosity_training_to_gpu.sh

# Mistral training with operatives-last processing
./scripts/mistral/deploy_mistral_to_g5.sh
```

## Repository Structure

```
├── scripts/
│   ├── qwen/                    # Qwen model training pipeline
│   ├── mistral/                 # Mistral model training pipeline  
│   ├── corpus-generation/       # Domain-specific data collection
│   └── monitoring/              # Training progress tracking
├── data/
│   ├── corpus/                  # Pre-built training datasets
│   ├── collectors/              # Data processing utilities
│   └── validation/              # Quality assurance pipelines
├── infrastructure/              # AWS deployment automation
├── training/                    # QLora trainers and frameworks
├── config/                      # Hardware-optimized configurations
└── tests/                       # Comprehensive test coverage
```

## Corpus Collection

### Supported Domains

- **IAC/DevOps** - Infrastructure as Code, CI/CD, containerization
- **Policy Analysis** - Government policy, insurance, academic research  
- **Security/Compliance** - Cybersecurity frameworks, compliance standards
- **NSFW Content** - Adult content classification and moderation

### Usage

```bash
# Collect domain-specific corpus
./scripts/corpus-generation/iac-devops-corpus/corpus-builders/create_final_iac_corpus.py

# Validate corpus quality
./data/validation/universal_validation_pipeline.py
```

## Training Pipelines

### Qwen Pipeline
- **Golden Config**: Optimized for Claude.md methodology compliance
- **Hardware**: g5.xlarge minimum, g5.2xlarge+ recommended
- **Specialization**: IAC/DevOps, code generation, system prompts

[→ Qwen Training Guide](./scripts/qwen/QWEN_TRAINING_GUIDE.md)

### Mistral Pipeline  
- **Operatives-Last Processing**: Handles 3M+ file collections efficiently
- **Hardware**: g5.2xlarge minimum for stable training
- **Specialization**: Policy analysis, multi-domain reasoning

[→ Mistral Training Guide](./scripts/mistral/MISTRAL_TRAINING_GUIDE.md)

## Infrastructure

### One-Shot Deployment

```bash
# Deploy training instance with automatic setup
./infrastructure/auto-deploy-mistral.sh

# Setup QLora training environment
./infrastructure/setup_qlora_instance.sh
```

### Hardware Configurations

| Instance Type | vCPUs | Memory | GPU | Best For |
|---------------|-------|--------|-----|----------|
| g5.xlarge | 4 | 16GB | 1x A10G | Development, small models |
| g5.2xlarge | 8 | 32GB | 1x A10G | Production training |
| g5.4xlarge | 16 | 64GB | 1x A10G | Large model fine-tuning |

## Monitoring

Real-time training progress tracking with S3 integration:

```bash
# Monitor active training runs
./scripts/monitoring/monitor_training.py

# Validate model deployment
./scripts/monitoring/validate_deployment.py
```

## Development

### Testing
```bash
# Run comprehensive test suite
pytest tests/

# Validate training configurations
./scripts/qwen/validate_qwen_styles.py
./scripts/mistral/validate_mistral_golden_config.py
```

### Contributing

1. Follow **CLAUDE.md** methodology: Explore → Plan → Code → Commit
2. All training data must be from authentic, real-world sources
3. Maintain comprehensive test coverage
4. Hardware configs must be validated across instance types

---

**Built by [Asoba](https://asoba.cloud) for production LLM training at scale.**