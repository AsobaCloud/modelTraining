#!/usr/bin/env python3
"""
Validation script for the monitored training deployment process.
Tests that all required files are copied and paths resolve correctly.
"""

import os
import subprocess
import sys
import tempfile
import json
from pathlib import Path

class DeploymentValidator:
    def __init__(self):
        self.errors = []
        self.warnings = []
        
    def log_error(self, msg):
        self.errors.append(f"❌ ERROR: {msg}")
        
    def log_warning(self, msg):
        self.warnings.append(f"⚠️  WARNING: {msg}")
        
    def log_success(self, msg):
        print(f"✅ {msg}")
        
    def validate_local_files(self):
        """Validate all required files exist locally"""
        print("\n🔍 Validating local files...")
        
        required_files = {
            "scripts/qwen/run_unsloth_training_monitored.sh": "Main deployment script",
            "scripts/qwen/train_qwen3_14b_optimal_sft": "Training script",
            "scripts/monitoring/training_monitor.py": "Monitoring module",
            "scripts/monitoring/s3_model_uploader.py": "S3 uploader",
            "scripts/monitoring/monitor_training.py": "Progress viewer",
            ".env": "Environment configuration"
        }
        
        for file, desc in required_files.items():
            if os.path.exists(file):
                self.log_success(f"Found {desc}: {file}")
            else:
                self.log_error(f"Missing {desc}: {file}")
                
    def validate_deployment_script(self):
        """Analyze deployment script for missing functionality"""
        print("\n🔍 Analyzing deployment script...")
        
        script_path = "scripts/qwen/run_unsloth_training_monitored.sh"
        if not os.path.exists(script_path):
            self.log_error(f"Cannot find {script_path}")
            return
            
        with open(script_path, 'r') as f:
            content = f.read()
            
        # Check for data file copying
        import re
        
        if "verbosity_pairs_qwen_chat_v2_1600_nosys_mixed.jsonl" in content:
            if re.search(r'scp.*verbosity_pairs_qwen_chat_v2_1600_nosys_mixed\.jsonl', content, re.DOTALL):
                self.log_success("Script copies verbosity data")
            else:
                self.log_error("Script checks for verbosity data but doesn't copy it")
                
        if "final_enhanced_iac_corpus.jsonl" in content:
            if re.search(r'scp.*final_enhanced_iac_corpus\.jsonl', content, re.DOTALL):
                self.log_success("Script copies IaC data")
            else:
                self.log_error("Script checks for IaC data but doesn't copy it")
                
        # Check for proper error handling
        if "set -euo pipefail" in content:
            self.log_success("Script has proper error handling")
        else:
            self.log_warning("Script missing 'set -euo pipefail'")
            
    def validate_data_files(self):
        """Check if required data files exist locally"""
        print("\n🔍 Checking for data files...")
        
        # Check multiple possible locations
        data_files = [
            ("verbosity_pairs_qwen_chat_v2_1600_nosys_mixed.jsonl", [
                ".",
                "data/",
                "scripts/qwen/",
                "../",
                "../../"
            ]),
            ("final_enhanced_iac_corpus.jsonl", [
                "data/comprehensive_iac_corpus/",
                "data/",
                ".",
                "scripts/qwen/data/",
                "../data/",
                "../../data/"
            ])
        ]
        
        for filename, paths in data_files:
            found = False
            for path in paths:
                full_path = os.path.join(path, filename)
                if os.path.exists(full_path):
                    self.log_success(f"Found {filename} at {full_path}")
                    found = True
                    break
            if not found:
                self.log_error(f"Cannot find {filename} in any expected location")
                
    def validate_remote_paths(self):
        """Analyze how paths will resolve on remote system"""
        print("\n🔍 Analyzing remote path resolution...")
        
        # Simulate remote directory structure
        remote_structure = {
            "~/qwen_training/": "Deployment target",
            "~/qwen_training/monitoring/": "Monitoring scripts",
            "~/qwen_training/data/": "Data directory (if created)",
            "~/": "Remote home"
        }
        
        print("Expected remote structure:")
        for path, desc in remote_structure.items():
            print(f"  {path} - {desc}")
            
        # Check if script handles copied location correctly
        script_path = "scripts/qwen/run_unsloth_training_monitored.sh"
        if os.path.exists(script_path):
            with open(script_path, 'r') as f:
                content = f.read()
                
            if "$SCRIPT_DIR/monitoring" in content:
                self.log_success("Script handles copied monitoring location")
            
            if "MONITOR_PATH" in content:
                self.log_success("Script uses MONITOR_PATH variable for flexibility")
                
    def generate_fixes(self):
        """Generate script to fix identified issues"""
        print("\n🔧 Generating fixes...")
        
        fixes = []
        
        # Check if we need to add data file copying
        script_path = "scripts/qwen/run_unsloth_training_monitored.sh"
        if os.path.exists(script_path):
            with open(script_path, 'r') as f:
                content = f.read()
                
            if 'scp.*verbosity_pairs' not in content:
                fixes.append("""
    # Copy data files
    echo "Copying data files..."
    
    # Find and copy verbosity data
    if [ -f "verbosity_pairs_qwen_chat_v2_1600_nosys_mixed.jsonl" ]; then
        scp -i "$SSH_KEY" "verbosity_pairs_qwen_chat_v2_1600_nosys_mixed.jsonl" "$REMOTE_USER@$REMOTE_HOST:~/qwen_training/"
    elif [ -f "../verbosity_pairs_qwen_chat_v2_1600_nosys_mixed.jsonl" ]; then
        scp -i "$SSH_KEY" "../verbosity_pairs_qwen_chat_v2_1600_nosys_mixed.jsonl" "$REMOTE_USER@$REMOTE_HOST:~/qwen_training/"
    else
        echo "WARNING: Cannot find verbosity data file to copy"
    fi
    
    # Find and copy IaC data  
    ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" "mkdir -p ~/qwen_training/data"
    if [ -f "data/final_enhanced_iac_corpus.jsonl" ]; then
        scp -i "$SSH_KEY" "data/final_enhanced_iac_corpus.jsonl" "$REMOTE_USER@$REMOTE_HOST:~/qwen_training/data/"
    elif [ -f "../data/final_enhanced_iac_corpus.jsonl" ]; then
        scp -i "$SSH_KEY" "../data/final_enhanced_iac_corpus.jsonl" "$REMOTE_USER@$REMOTE_HOST:~/qwen_training/data/"
    else
        echo "WARNING: Cannot find IaC data file to copy"
    fi""")
                
        if fixes:
            print("\nRequired fixes identified:")
            for fix in fixes:
                print(fix)
        else:
            print("No fixes required")
            
    def run_validation(self):
        """Run all validation checks"""
        print("🚀 Deployment Validation Report")
        print("=" * 60)
        
        self.validate_local_files()
        self.validate_deployment_script()
        self.validate_data_files()
        self.validate_remote_paths()
        self.generate_fixes()
        
        print("\n📊 Summary")
        print("=" * 60)
        
        if self.errors:
            print(f"\n{len(self.errors)} errors found:")
            for error in self.errors:
                print(error)
                
        if self.warnings:
            print(f"\n{len(self.warnings)} warnings found:")
            for warning in self.warnings:
                print(warning)
                
        if not self.errors:
            print("\n✅ Deployment script validation passed!")
            return 0
        else:
            print("\n❌ Deployment script has issues that need to be fixed")
            return 1

if __name__ == "__main__":
    validator = DeploymentValidator()
    sys.exit(validator.run_validation())