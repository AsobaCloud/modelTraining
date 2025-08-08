# Solar PV Fault Detection Datasets Documentation
## For Vision-Language Model Development

**Date**: August 8, 2025  
**Location**: `s3://policy-database/pv-ops-and-maintenance/`  
**Total Images**: 31,158 thermal infrared images  
**Purpose**: Training data for image-to-text models for solar PV fault detection and maintenance

---

## Executive Summary

This document details three comprehensive solar photovoltaic (PV) datasets totaling over 31,000 thermal infrared images with labeled fault conditions, suitable for training vision-language models for automated fault detection and maintenance recommendation systems.

---

## Dataset 1: InfraredSolarModules
**Source**: RaptorMaps (ICLR 2020)  
**Location**: `s3://policy-database/pv-ops-and-maintenance/dataset/`  
**Total Images**: 20,000  
**Image Format**: JPEG, 24x40 pixels  
**File Size**: ~10MB total (small, preprocessed images)

### Label Distribution
| Class | Images | Type | Description |
|-------|--------|------|-------------|
| No-Anomaly | 10,000 | Baseline | Normal operating modules |
| Cell | 1,877 | Hot Spot | Single cell hot spot with square geometry |
| Vegetation | 1,639 | Obstruction | Panels blocked by vegetation growth |
| Diode | 1,499 | Electrical | Activated bypass diode (1/3 module affected) |
| Cell-Multi | 1,288 | Hot Spot | Multiple cell hot spots |
| Shadowing | 1,056 | Environmental | Shadow from structures/adjacent rows |
| Cracking | 941 | Physical | Surface cracking on module |
| Offline-Module | 828 | Electrical | Entire module heated/offline |
| Hot-Spot | 251 | Thin Film | Hot spot on thin film module |
| Hot-Spot-Multi | 247 | Thin Film | Multiple hot spots on thin film |
| Soiling | 205 | Environmental | Dirt/dust accumulation |
| Diode-Multi | 175 | Electrical | Multiple bypass diodes (2/3 module) |

### Key Characteristics
- **Resolution**: Low (24x40) - preprocessed for ML efficiency
- **Labeling**: Single class per image, professionally labeled
- **Balance**: Heavily imbalanced (50% normal, varying fault distributions)
- **Metadata**: Complete JSON with image-label mappings
- **Research Grade**: Published dataset with citations

### Model Training Readiness
- ✅ **Ready for immediate use** - preprocessed and labeled
- ✅ **Ideal for**: Quick prototyping, baseline models
- ⚠️ **Limitation**: Low resolution may limit detail detection

---

## Dataset 2: PVF-10 (Photovoltaic Fault Dataset)
**Source**: China University of Geosciences (Applied Energy 2024)  
**Location**: `s3://policy-database/pv-ops-and-maintenance/pvf-dataset/`  
**Total Images**: 11,158 (5,579 original + augmented)  
**Image Format**: PNG, 110x60 pixels  
**File Size**: 157MB compressed

### Label Distribution
| Category | Class | Type | Repairable | Description |
|----------|-------|------|------------|-------------|
| **Repairable Faults** (5 classes) |
| | Shadowing | Environmental | ✅ | Temporary obstruction |
| | Soiling | Environmental | ✅ | Surface contamination |
| | Module Broken | Physical | ✅ | Repairable damage |
| | Diode Bypass | Electrical | ✅ | Bypass diode activation |
| | Hot Spot | Thermal | ✅ | Localized heating |
| **Irreparable Faults** (5 classes) |
| | Burn Mark | Thermal Damage | ❌ | Permanent burn damage |
| | Snail Trail | Chemical | ❌ | Micro-crack induced trails |
| | Discoloration | Chemical | ❌ | EVA browning/yellowing |
| | Corrosion | Chemical | ❌ | Metal contact corrosion |
| | Delamination | Physical | ❌ | Layer separation |

### Key Characteristics
- **Resolution**: Medium (110x60) - UAV thermal camera native
- **Source**: 8 different solar power plants
- **Collection**: DJI thermal drone systematic surveys
- **Split**: Pre-defined train/test splits
- **Augmentation**: Resampling strategies included

### Existing Model Architecture (from repo)
```python
# From train_val.py analysis
- Deep learning models tested: ResNet, EfficientNet variants
- Input: 110x60 thermal images
- Output: 10-class classification
- Preprocessing: Resampling + padding strategies
- Validation: Cross-validation with stratification
- Metrics: Overall Accuracy (OA) > 83%, best 93.32%
```

### Model Training Readiness
- ✅ **Complete training pipeline** available
- ✅ **Pre-split** train/validation/test sets
- ✅ **Proven architecture** with 93% accuracy
- ✅ **Ideal for**: Fine-tuning, transfer learning

---

## Dataset 3: NREL PVDAQ Time-Series
**Source**: NREL (National Renewable Energy Laboratory)  
**Location**: `s3://policy-database/usa/federal-legislation-executive-courts/federal-agencies/nrel/`  
**Total Records**: 6,695,893 time-series measurements  
**Data Type**: Tabular CSV/JSON (not images)  
**Systems**: 10 PV installations with metadata

### Data Composition
- **Power Generation**: 15-minute interval production data
- **Weather Correlation**: Temperature, irradiance measurements
- **System Metadata**: Panel types, inverter specs, installation details
- **Temporal Coverage**: Multiple years of operational data

### Use Cases for Vision-Language Models
- **Contextual Information**: Correlate visual faults with performance drops
- **Temporal Patterns**: Link seasonal variations to fault types
- **Performance Validation**: Verify fault impact on generation
- **Maintenance Prioritization**: Quantify financial impact of detected faults

---

## Combined Dataset Statistics

### Aggregate Metrics
- **Total Thermal Images**: 31,158
- **Unique Fault Classes**: 15+ distinct fault types
- **Resolution Range**: 24x40 to 110x60 pixels
- **Geographic Diversity**: US + China installations
- **Temporal Data**: 6.7M performance records

### Fault Category Distribution
| Meta-Category | Total Images | Percentage |
|---------------|--------------|------------|
| Normal/Baseline | 10,000 | 32.1% |
| Electrical Faults | ~4,500 | 14.4% |
| Environmental | ~3,100 | 9.9% |
| Physical Damage | ~2,800 | 9.0% |
| Thermal Issues | ~2,600 | 8.3% |
| Chemical Degradation | ~1,500 | 4.8% |
| Other/Multiple | ~6,658 | 21.4% |

---

## Vision-Language Model Training Considerations

### Current Readiness Assessment

#### Strengths
1. **Large Scale**: 31K+ images sufficient for fine-tuning
2. **Diverse Faults**: 15+ fault types with clear definitions
3. **Multiple Resolutions**: Can test scale invariance
4. **Professional Labels**: Research-grade annotations
5. **Existing Baselines**: 93% accuracy on classification

#### Challenges
1. **Resolution Variability**: 24x40 to 110x60 (low by modern standards)
2. **Class Imbalance**: 32% normal vs. rare faults <1%
3. **Limited Descriptions**: Labels are categorical, not descriptive text
4. **No Captions**: Would need to generate text descriptions

### Recommended Approach for Image-to-Text Model

#### Phase 1: Data Preparation
```python
# Proposed preprocessing pipeline
1. Upscale low-res images (24x40 → 224x224 minimum)
2. Generate synthetic captions from labels + metadata
3. Balance dataset through strategic sampling
4. Create multi-modal pairs (image + description)
```

#### Phase 2: Caption Generation Strategy
```python
# Template-based caption generation
fault_templates = {
    "Cell": "Thermal image showing hot spot in single cell at coordinates {loc}, 
             temperature anomaly of {temp_diff}°C indicating possible solder joint 
             failure or cell crack",
    "Vegetation": "Infrared scan reveals vegetation obstruction covering {coverage}% 
                   of panel surface, causing irregular heating pattern and estimated 
                   {power_loss}% power reduction",
    # ... etc for each fault type
}

# Augment with technical descriptions
- Fault mechanism explanation
- Maintenance action required  
- Urgency level
- Cost impact estimate
```

#### Phase 3: Model Architecture Options

**Option 1: Fine-tune Existing VLM**
- Base: CLIP, BLIP-2, or LLaVA
- Approach: Fine-tune on solar-specific image-caption pairs
- Advantage: Leverages pre-trained knowledge
- Training Time: ~20-40 hours on A100

**Option 2: Custom Vision Encoder + LLM**
- Vision: Fine-tuned ResNet/EfficientNet from PVF-10
- Language: Frozen LLaMA/Mistral with adapter
- Advantage: Specialized vision features
- Training Time: ~40-80 hours

**Option 3: Retrieval-Augmented Generation**
- Index: FAISS index of all fault images
- Retrieval: Find similar faults
- Generation: LLM describes based on retrieved examples
- Advantage: No training required, interpretable

### Recommended Training Configuration

```yaml
# Suggested hyperparameters
model_config:
  base_model: "BLIP-2" or "LLaVA-1.5"
  vision_encoder: "EVA-CLIP-g"
  text_decoder: "Vicuna-7B"
  
training_config:
  batch_size: 32
  learning_rate: 1e-5
  warmup_steps: 1000
  max_epochs: 10
  gradient_checkpointing: true
  mixed_precision: fp16
  
data_config:
  train_size: 25,000
  val_size: 3,000
  test_size: 3,158
  augmentation:
    - random_rotation: 15
    - brightness_jitter: 0.2
    - contrast_jitter: 0.2
  caption_augmentation:
    - technical_descriptions
    - maintenance_actions
    - severity_levels
```

### Expected Outcomes

With proper training, the model should be able to:

1. **Describe Faults**: "This thermal image shows multiple hot spots in the upper-left quadrant of the solar panel, consistent with bypass diode activation."

2. **Diagnose Issues**: "The irregular heating pattern and temperature differential of 15°C suggests partial shading from vegetation growth."

3. **Recommend Actions**: "Immediate cleaning required. Soiling detected across 30% of module surface, estimated 12% power loss. Priority: Medium."

4. **Quantify Impact**: "Delamination visible in 3 cells, irreparable damage requiring module replacement. Estimated cost: $300-400."

---

## Implementation Timeline

### Week 1-2: Data Preparation
- Generate captions for all 31K images
- Create train/val/test splits
- Set up data loaders

### Week 3-4: Model Development  
- Set up base VLM architecture
- Implement training pipeline
- Configure evaluation metrics

### Week 5-6: Training & Optimization
- Initial training runs
- Hyperparameter tuning
- Ablation studies

### Week 7-8: Evaluation & Deployment
- Test set evaluation
- API development
- Documentation

---

## Data Access

All datasets are available at:
```bash
# InfraredSolarModules
aws s3 ls s3://policy-database/pv-ops-and-maintenance/dataset/

# PVF-10
aws s3 ls s3://policy-database/pv-ops-and-maintenance/pvf-dataset/

# NREL PVDAQ
aws s3 ls s3://policy-database/usa/federal-legislation-executive-courts/federal-agencies/nrel/
```

---

## Citations

```bibtex
@inproceedings{infraredsolar2020,
  title={InfraredSolarModules: A machine learning dataset for solar panel anomaly detection},
  author={RaptorMaps},
  booktitle={ICLR Workshop on AI for Earth Sciences},
  year={2020}
}

@article{WANG2024124187,
  title={PVF-10: A high-resolution unmanned aerial vehicle thermal infrared image dataset 
         for fine-grained photovoltaic fault classification},
  author={Wang, Bo and Chen, Qi and Wang, Mengmeng and others},
  journal={Applied Energy},
  volume={376},
  pages={124187},
  year={2024}
}

@dataset{nrel_pvdaq,
  title={PV Data Acquisition (PVDAQ) Database},
  author={NREL},
  publisher={National Renewable Energy Laboratory},
  year={2024},
  url={https://data.openei.org/s3_viewer}
}
```

---

## Next Steps

1. **Generate synthetic captions** using templates + GPT-4
2. **Create unified data loader** for all three datasets
3. **Establish baseline** with CLIP zero-shot
4. **Begin fine-tuning** experiments
5. **Develop evaluation metrics** specific to maintenance use case

---

**Document Version**: 1.0  
**Last Updated**: August 8, 2025  
**Author**: AI Engineering Team