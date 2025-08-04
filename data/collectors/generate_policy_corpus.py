#!/usr/bin/env python3
"""
Policy Analysis Training Corpus Generator
Convert processed PDF data to training-ready JSONL format for Mistral QLoRA
Following complete_mistral_qlora_training_guide.md format specifications
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PolicyCorpusGenerator:
    """Generate policy analysis training corpus in JSONL format"""
    
    def __init__(self):
        """Initialize corpus generator"""
        self.corpus_stats = {
            "total_examples": 0,
            "domain_distribution": {},
            "type_distribution": {},
            "quality_distribution": {},
            "average_prompt_length": 0,
            "average_completion_length": 0
        }
        
        # Quality thresholds
        self.min_prompt_length = 10
        self.max_prompt_length = 500
        self.min_completion_length = 50
        self.max_completion_length = 3000
        self.min_quality_score = 0.4

    def validate_training_example(self, example: Dict[str, Any]) -> bool:
        """Validate training example meets quality standards"""
        try:
            # Required fields
            if not all(key in example for key in ["prompt", "completion"]):
                return False
            
            prompt = example["prompt"].strip()
            completion = example["completion"].strip()
            
            # Length constraints
            if not (self.min_prompt_length <= len(prompt) <= self.max_prompt_length):
                return False
            
            if not (self.min_completion_length <= len(completion) <= self.max_completion_length):
                return False
            
            # Quality score threshold
            metadata = example.get("metadata", {})
            quality_score = metadata.get("quality_score", 0)
            if quality_score < self.min_quality_score:
                return False
            
            # Content quality checks
            if not self.check_content_quality(prompt, completion):
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Validation error: {e}")
            return False

    def check_content_quality(self, prompt: str, completion: str) -> bool:
        """Check content quality of prompt-completion pair"""
        # Avoid repetitive or low-quality content
        prompt_lower = prompt.lower()
        completion_lower = completion.lower()
        
        # Check for meaningful policy-related content
        policy_keywords = [
            "policy", "analysis", "legislation", "regulatory", "government",
            "economic", "social", "environmental", "political", "legal",
            "implementation", "recommendations", "impact", "assessment"
        ]
        
        has_policy_content = any(keyword in completion_lower for keyword in policy_keywords)
        if not has_policy_content:
            return False
        
        # Avoid extremely repetitive text
        unique_words = len(set(completion.split()))
        total_words = len(completion.split())
        if unique_words / total_words < 0.3:  # Less than 30% unique words
            return False
        
        # Check for proper structure in completion
        has_structure = any(marker in completion for marker in [
            "##", "###", "1.", "2.", "•", "-", ":", "analysis", "summary", "recommendations"
        ])
        
        return has_structure

    def format_for_mistral_training(self, example: Dict[str, Any]) -> Dict[str, str]:
        """Format example for Mistral QLoRA training following guide format"""
        prompt = example["prompt"].strip()
        completion = example["completion"].strip()
        
        # Use the format from complete_mistral_qlora_training_guide.md
        formatted_example = {
            "prompt": prompt,
            "completion": completion
        }
        
        return formatted_example

    def process_pdf_results(self, processing_file: str) -> List[Dict[str, Any]]:
        """Process PDF processing results into training examples"""
        logger.info(f"Loading PDF processing results from {processing_file}")
        
        try:
            with open(processing_file, 'r', encoding='utf-8') as f:
                processing_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading processing file {processing_file}: {e}")
            return []
        
        all_examples = []
        
        # Extract training examples from processing results
        for pdf_result in processing_data.get("processed_pdfs", []):
            if pdf_result.get("success", False):
                examples = pdf_result.get("training_examples", [])
                all_examples.extend(examples)
        
        # Also check for direct training examples
        direct_examples = processing_data.get("all_training_examples", [])
        all_examples.extend(direct_examples)
        
        logger.info(f"Found {len(all_examples)} raw training examples")
        return all_examples

    def generate_corpus(self, processing_file: str, output_file: str, max_examples: int = 2000) -> Dict[str, Any]:
        """Generate complete training corpus from processing results"""
        logger.info(f"🚀 Generating policy analysis training corpus")
        logger.info(f"Processing file: {processing_file}")
        logger.info(f"Output file: {output_file}")
        logger.info(f"Max examples: {max_examples}")
        
        generation_results = {
            "start_time": datetime.now().isoformat(),
            "source_file": processing_file,
            "output_file": output_file,
            "generation_stats": {},
            "corpus_quality_report": {},
            "errors": []
        }
        
        # Load raw examples
        raw_examples = self.process_pdf_results(processing_file)
        
        if not raw_examples:
            error_msg = "No training examples found in processing file"
            logger.error(error_msg)
            generation_results["errors"].append(error_msg)
            return generation_results
        
        # Validate and format examples
        valid_examples = []
        validation_stats = {
            "total_raw": len(raw_examples),
            "validation_passed": 0,
            "validation_failed": 0,
            "format_errors": 0
        }
        
        for example in raw_examples:
            try:
                if self.validate_training_example(example):
                    formatted_example = self.format_for_mistral_training(example)
                    valid_examples.append({
                        "formatted": formatted_example,
                        "metadata": example.get("metadata", {})
                    })
                    validation_stats["validation_passed"] += 1
                else:
                    validation_stats["validation_failed"] += 1
            except Exception as e:
                validation_stats["format_errors"] += 1
                logger.debug(f"Format error: {e}")
        
        logger.info(f"Validation: {validation_stats['validation_passed']}/{validation_stats['total_raw']} examples passed")
        
        # Limit examples if needed
        if max_examples and len(valid_examples) > max_examples:
            # Sort by quality score for better selection
            valid_examples.sort(
                key=lambda x: x["metadata"].get("quality_score", 0), 
                reverse=True
            )
            valid_examples = valid_examples[:max_examples]
            logger.info(f"Limited to top {max_examples} examples by quality")
        
        # Generate corpus statistics
        self.analyze_corpus_quality(valid_examples)
        
        # Write corpus to JSONL file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                for example_data in valid_examples:
                    formatted_example = example_data["formatted"]
                    json.dump(formatted_example, f, ensure_ascii=False)
                    f.write('\n')
            
            logger.info(f"✅ Corpus saved to {output_file}")
            
        except Exception as e:
            error_msg = f"Error saving corpus to {output_file}: {e}"
            logger.error(error_msg)
            generation_results["errors"].append(error_msg)
            return generation_results
        
        # Finalize results
        generation_results.update({
            "end_time": datetime.now().isoformat(),
            "generation_stats": {
                "raw_examples": len(raw_examples),
                "valid_examples": len(valid_examples),
                "final_corpus_size": len(valid_examples),
                "validation_stats": validation_stats
            },
            "corpus_quality_report": self.corpus_stats
        })
        
        logger.info("📊 Corpus Generation Summary:")
        logger.info(f"  Raw examples: {len(raw_examples)}")
        logger.info(f"  Valid examples: {len(valid_examples)}")
        logger.info(f"  Final corpus size: {len(valid_examples)}")
        logger.info(f"  Average prompt length: {self.corpus_stats['average_prompt_length']:.1f}")
        logger.info(f"  Average completion length: {self.corpus_stats['average_completion_length']:.1f}")
        
        return generation_results

    def analyze_corpus_quality(self, examples: List[Dict[str, Any]]):
        """Analyze corpus quality and generate statistics"""
        if not examples:
            return
        
        prompt_lengths = []
        completion_lengths = []
        quality_scores = []
        domains = []
        types = []
        
        for example_data in examples:
            example = example_data["formatted"]
            metadata = example_data["metadata"]
            
            prompt_lengths.append(len(example["prompt"]))
            completion_lengths.append(len(example["completion"]))
            quality_scores.append(metadata.get("quality_score", 0))
            
            # Domain distribution
            domain = metadata.get("domain", metadata.get("domains", ["general"]))
            if isinstance(domain, list):
                domains.extend(domain)
            else:
                domains.append(domain)
            
            # Type distribution  
            example_type = metadata.get("type", "general")
            types.append(example_type)
        
        # Calculate statistics
        self.corpus_stats.update({
            "total_examples": len(examples),
            "average_prompt_length": sum(prompt_lengths) / len(prompt_lengths),
            "average_completion_length": sum(completion_lengths) / len(completion_lengths),
            "average_quality_score": sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            "domain_distribution": self.count_distribution(domains),
            "type_distribution": self.count_distribution(types),
            "quality_distribution": {
                "min": min(quality_scores) if quality_scores else 0,
                "max": max(quality_scores) if quality_scores else 0,
                "median": sorted(quality_scores)[len(quality_scores)//2] if quality_scores else 0
            }
        })

    def count_distribution(self, items: List[str]) -> Dict[str, int]:
        """Count distribution of items"""
        distribution = {}
        for item in items:
            distribution[item] = distribution.get(item, 0) + 1
        return dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))

    def validate_corpus_file(self, corpus_file: str) -> Dict[str, Any]:
        """Validate generated corpus file"""
        logger.info(f"🔍 Validating corpus file: {corpus_file}")
        
        validation_results = {
            "file_exists": False,
            "line_count": 0,
            "valid_json_lines": 0,
            "invalid_json_lines": 0,
            "sample_examples": [],
            "validation_errors": []
        }
        
        try:
            if not Path(corpus_file).exists():
                validation_results["validation_errors"].append("Corpus file does not exist")
                return validation_results
            
            validation_results["file_exists"] = True
            
            # Read and validate each line
            with open(corpus_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    validation_results["line_count"] += 1
                    
                    try:
                        example = json.loads(line)
                        
                        # Validate required fields
                        if "prompt" in example and "completion" in example:
                            validation_results["valid_json_lines"] += 1
                            
                            # Collect sample examples
                            if len(validation_results["sample_examples"]) < 3:
                                validation_results["sample_examples"].append({
                                    "line": line_num,
                                    "prompt": example["prompt"][:100] + "..." if len(example["prompt"]) > 100 else example["prompt"],
                                    "completion": example["completion"][:100] + "..." if len(example["completion"]) > 100 else example["completion"]
                                })
                        else:
                            validation_results["invalid_json_lines"] += 1
                            validation_results["validation_errors"].append(f"Line {line_num}: Missing required fields")
                    
                    except json.JSONDecodeError as e:
                        validation_results["invalid_json_lines"] += 1
                        validation_results["validation_errors"].append(f"Line {line_num}: JSON decode error: {e}")
            
            # Calculate validation rate
            total_lines = validation_results["valid_json_lines"] + validation_results["invalid_json_lines"]
            validation_rate = validation_results["valid_json_lines"] / total_lines if total_lines > 0 else 0
            
            logger.info(f"✅ Corpus validation complete:")
            logger.info(f"  Total lines: {validation_results['line_count']}")
            logger.info(f"  Valid examples: {validation_results['valid_json_lines']}")
            logger.info(f"  Validation rate: {validation_rate:.1%}")
            
            if validation_results["validation_errors"]:
                logger.warning(f"  Validation errors: {len(validation_results['validation_errors'])}")
                for error in validation_results["validation_errors"][:5]:
                    logger.warning(f"    {error}")
            
        except Exception as e:
            error_msg = f"Error validating corpus file: {e}"
            validation_results["validation_errors"].append(error_msg)
            logger.error(error_msg)
        
        return validation_results

    def save_generation_report(self, results: Dict[str, Any], report_file: str):
        """Save corpus generation report"""
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"📊 Generation report saved to: {report_file}")
            
        except Exception as e:
            logger.error(f"Error saving report to {report_file}: {e}")


def main():
    """Main execution function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python generate_policy_corpus.py <processing_results.json> [output_file] [max_examples]")
        sys.exit(1)
    
    processing_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "policy_analysis_corpus.jsonl"
    max_examples = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    
    logger.info(f"🚀 Policy Corpus Generation Started")
    logger.info(f"Processing file: {processing_file}")
    logger.info(f"Output file: {output_file}")
    logger.info(f"Max examples: {max_examples}")
    
    # Initialize generator
    generator = PolicyCorpusGenerator()
    
    # Generate corpus
    results = generator.generate_corpus(processing_file, output_file, max_examples)
    
    # Validate corpus
    validation_results = generator.validate_corpus_file(output_file)
    results["corpus_validation"] = validation_results
    
    # Save generation report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"policy_corpus_generation_{timestamp}.json"
    generator.save_generation_report(results, report_file)
    
    # Print final summary
    if "generation_stats" in results:
        stats = results["generation_stats"]
        logger.info("🎯 Final Generation Summary:")
        logger.info(f"  📄 Raw examples: {stats['raw_examples']}")
        logger.info(f"  ✅ Valid examples: {stats['valid_examples']}")
        logger.info(f"  📚 Final corpus size: {stats['final_corpus_size']}")
        logger.info(f"  💾 Corpus file: {output_file}")
        logger.info(f"  📊 Report file: {report_file}")
        
        # Show validation results
        if validation_results["valid_json_lines"] > 0:
            logger.info(f"  🔍 Validation: {validation_results['valid_json_lines']} valid examples")
    
    return results


if __name__ == "__main__":
    results = main()