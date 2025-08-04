#!/usr/bin/env python3
"""
Monitor Smart Collectors Progress and Combine Results

Monitors the progress of all smart collectors and combines their outputs
into a unified training corpus for Qwen fine-tuning.
"""

import json
import time
import subprocess
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart_collector_monitor")

class SmartCollectorMonitor:
    """Monitor and combine smart collector results"""
    
    def __init__(self):
        self.corpus_dir = Path("/home/shingai/sort/deployments/data/corpus")
        self.expected_files = [
            "smart_code_generation_corpus.jsonl",
            "smart_technical_documentation_corpus.jsonl", 
            "smart_devops_automation_corpus.jsonl",
            "smart_data_engineering_corpus.jsonl",
            "smart_system_architecture_corpus.jsonl",
            "smart_asobacode_mcp_corpus.jsonl"
        ]
        
        # Original corpus files to include
        self.existing_files = [
            "combined_iac_corpus.jsonl",  # 2,193 IaC examples
            "asobacode_mcp_corpus.jsonl"  # 80 local MCP examples (will be replaced by smart version)
        ]
    
    def check_processes_running(self) -> List[str]:
        """Check which smart collector processes are still running"""
        try:
            result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            running_collectors = []
            
            for line in result.stdout.split('\n'):
                if 'smart_' in line and 'collector.py' in line and 'python3' in line:
                    # Extract collector name
                    for expected in self.expected_files:
                        collector_name = expected.replace('_corpus.jsonl', '_collector.py')
                        if collector_name in line:
                            running_collectors.append(expected.replace('_corpus.jsonl', ''))
            
            return running_collectors
        except Exception as e:
            logger.error(f"Error checking processes: {e}")
            return []
    
    def check_completed_files(self) -> Dict[str, int]:
        """Check which corpus files have been created and their sizes"""
        completed = {}
        
        for filename in self.expected_files:
            filepath = self.corpus_dir / filename
            if filepath.exists():
                try:
                    with open(filepath, 'r') as f:
                        line_count = sum(1 for _ in f)
                    completed[filename] = line_count
                except Exception as e:
                    logger.warning(f"Could not read {filename}: {e}")
        
        return completed
    
    def combine_all_corpora(self) -> Dict[str, int]:
        """Combine all available corpora into unified training set"""
        logger.info("Combining all available corpora...")
        
        combined_examples = []
        domain_counts = {}
        
        # Add IaC examples (foundational 2,193 examples)
        iac_file = self.corpus_dir / "combined_iac_corpus.jsonl"
        if iac_file.exists():
            logger.info("Adding IaC examples...")
            with open(iac_file, 'r') as f:
                for line in f:
                    example = json.loads(line.strip())
                    combined_examples.append(example)
                    domain = example.get('domain', 'iac')
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        # Add smart collector results
        for filename in self.expected_files:
            filepath = self.corpus_dir / filename
            if filepath.exists():
                domain_name = filename.replace('smart_', '').replace('_corpus.jsonl', '')
                logger.info(f"Adding {domain_name} examples...")
                
                with open(filepath, 'r') as f:
                    for line in f:
                        try:
                            example = json.loads(line.strip())
                            combined_examples.append(example)
                            domain = example.get('domain', domain_name)
                            domain_counts[domain] = domain_counts.get(domain, 0) + 1
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping invalid JSON in {filename}: {e}")
        
        # Save combined corpus
        output_file = self.corpus_dir / "qwen_multi_domain_training_corpus.jsonl"
        logger.info(f"Saving combined corpus to {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in combined_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Combined corpus created with {len(combined_examples)} examples")
        
        return domain_counts
    
    def print_status_report(self):
        """Print comprehensive status report"""
        print("\n" + "="*60)
        print("SMART COLLECTOR STATUS REPORT")
        print("="*60)
        
        # Check running processes
        running = self.check_processes_running()
        print(f"\nRunning Collectors: {len(running)}")
        for collector in running:
            print(f"  ✓ {collector}")
        
        # Check completed files
        completed = self.check_completed_files()
        print(f"\nCompleted Collections: {len(completed)}")
        total_smart_examples = 0
        for filename, count in completed.items():
            domain = filename.replace('smart_', '').replace('_corpus.jsonl', '')
            print(f"  ✓ {domain}: {count} examples")
            total_smart_examples += count
        
        # Check existing files
        print(f"\nExisting Corpus Files:")
        for filename in self.existing_files:
            filepath = self.corpus_dir / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    count = sum(1 for _ in f)
                print(f"  ✓ {filename}: {count} examples")
        
        print(f"\nTotal Smart Collector Examples: {total_smart_examples}")
        print(f"Expected Target: ~6,000 examples across all domains")
        
        # Show progress percentage
        if total_smart_examples > 0:
            progress = (total_smart_examples / 6000) * 100
            print(f"Progress: {progress:.1f}% of target reached")
    
    def wait_for_completion_and_combine(self, check_interval: int = 60):
        """Wait for all collectors to complete and combine results"""
        logger.info("Monitoring smart collectors...")
        
        while True:
            running = self.check_processes_running()
            completed = self.check_completed_files()
            
            self.print_status_report()
            
            if not running:
                logger.info("All collectors completed! Combining results...")
                domain_counts = self.combine_all_corpora()
                
                print("\n" + "="*60)
                print("FINAL TRAINING CORPUS SUMMARY")
                print("="*60)
                
                total_examples = sum(domain_counts.values())
                print(f"Total Examples: {total_examples}")
                print("\nDomain Breakdown:")
                for domain, count in sorted(domain_counts.items()):
                    percentage = (count / total_examples) * 100
                    print(f"  {domain}: {count} examples ({percentage:.1f}%)")
                
                print(f"\nTraining corpus ready for Qwen fine-tuning!")
                print(f"Location: {self.corpus_dir}/qwen_multi_domain_training_corpus.jsonl")
                break
            
            logger.info(f"Waiting {check_interval}s... {len(running)} collectors still running")
            time.sleep(check_interval)

def main():
    """Main monitoring process"""
    monitor = SmartCollectorMonitor()
    
    # Show initial status
    monitor.print_status_report()
    
    # Wait for completion and combine results
    monitor.wait_for_completion_and_combine(check_interval=300)  # Check every 5 minutes

if __name__ == "__main__":
    main()