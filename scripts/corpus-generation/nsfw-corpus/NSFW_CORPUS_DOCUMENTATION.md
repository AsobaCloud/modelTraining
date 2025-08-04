# NSFW Corpus Documentation

## Dataset Location
**Primary Source**: S3 Bucket `s3://flux-dev-nsfw/`
- **Region**: us-west-2
- **Total Images**: 3,343 NSFW images
- **Format**: JPEG files with various resolutions
- **Access**: AWS CLI required (`aws s3 ls s3://flux-dev-nsfw/`)

## Training Status
✅ **Training Completed Successfully**
- **Steps**: 2000/2000 (100% complete)
- **Final Loss**: 0.0005
- **Training Time**: ~77.5 hours
- **LoRA Adapter**: `/home/ubuntu/flux_training/outputs/checkpoints/checkpoint-1500/adapter_model.safetensors`

## Dataset Structure
```
s3://flux-dev-nsfw/
├── 1000009390.JPEG
├── 1000013485.JPEG  
├── 1000014058.jpeg
├── 1983 Zaawaadi/
│   ├── 1983_001.jpg
│   └── 1983_002.jpg
├── Anissa Kate - Big Tits Curvy Wife.../
├── Blacked Kelly Collins.../
├── Kazumi & Nicole Doshi.../
└── [3,343 total files]
```

## Training Configuration
- **Base Model**: black-forest-labs/FLUX.1-dev
- **LoRA Rank**: 32, Alpha: 16, Dropout: 0.1
- **Resolution**: 1024x1024 (upscaled from original)
- **Batch Size**: 1 with 8 gradient accumulation steps
- **Learning Rate**: 1e-4 with 100 warmup steps

## Integration Status
❌ **Current Issue**: NSFW LoRA adapter not loading in server
- Server shows: `"lora_loaded": false, "variant": "base"`
- Expected: `"lora_loaded": true, "variant": "nsfw-trained"`

## Usage
To use the NSFW-trained model:
1. Load base FLUX.1-dev model
2. Apply LoRA adapter from checkpoint-1500
3. Generate with NSFW prompts

## Files
- **LoRA Weights**: `flux_training/outputs/checkpoints/checkpoint-1500/adapter_model.safetensors` (25MB)
- **Training Config**: `flux_training/flux_training_config.json`
- **Training Logs**: `flux_training/outputs/logs/production_final_20250720_105903.log`