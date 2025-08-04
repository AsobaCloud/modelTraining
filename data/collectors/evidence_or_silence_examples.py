#!/usr/bin/env python3
"""
Evidence-or-Silence Training Examples Generator
Create specific training examples demonstrating the evidence-or-silence principle
"""

import json
import logging
from typing import List, Dict, Any
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EvidenceOrSilenceExamples:
    """Generate training examples that demonstrate evidence-or-silence principle"""
    
    def __init__(self):
        """Initialize with example templates"""
        pass

    def create_insufficient_evidence_examples(self) -> List[Dict[str, Any]]:
        """Create examples where the model should respond with 'Insufficient evidence'"""
        examples = []
        
        insufficient_evidence_scenarios = [
            {
                "prompt": "What will be the exact economic impact of the proposed carbon tax in 2025?",
                "completion": """### Analysis:

**Insufficient evidence.**

To provide an authoritative analysis of the exact economic impact of a proposed carbon tax in 2025, the following primary sources would be required:

- **Statutory Text**: Specific legislation with tax rates, covered sectors, and implementation timeline
- **Regulatory Framework**: Implementing regulations defining covered emissions and compliance mechanisms  
- **Economic Modeling**: Official government economic impact assessments from agencies like EPA or CBO
- **Baseline Data**: Current sectoral emissions data and economic baseline projections

Without access to these verified primary sources, any specific quantitative projections would be speculative rather than evidence-based.

**Authority Assessment**: Insufficient primary source documentation
**Evidence Confidence**: 0.0 - No authoritative sources available for analysis""",
                "authority_type": "insufficient",
                "authority_score": 0.0,
                "evidence_confidence": 0.0
            },
            {
                "prompt": "How many jobs will the Green New Deal create in the transportation sector?",
                "completion": """### Analysis:

**Insufficient evidence.**

Authoritative analysis of job creation impacts requires:

- **Legislative Text**: Final enacted legislation with specific transportation provisions
- **Agency Analysis**: Official economic impact assessments from relevant departments (DOT, DOE, EPA)
- **Methodological Framework**: Government-approved economic modeling assumptions and parameters
- **Sectoral Data**: Baseline employment statistics and projected industry changes

Current analysis cannot proceed without verified government sources providing specific employment projections backed by established economic methodology.

**Authority Assessment**: Insufficient regulatory and analytical documentation
**Evidence Confidence**: 0.0 - No primary source employment projections available""",
                "authority_type": "insufficient", 
                "authority_score": 0.0,
                "evidence_confidence": 0.0
            },
            {
                "prompt": "What are the privacy implications of the new federal AI surveillance regulations?",
                "completion": """### Analysis:

**Insufficient evidence.**

Analysis of federal AI surveillance regulations requires access to:

- **Regulatory Text**: Specific CFR provisions defining AI surveillance requirements
- **Statutory Authority**: Underlying legal framework authorizing surveillance activities
- **Privacy Impact Assessments**: Official agency evaluations of privacy implications
- **Implementation Guidance**: Agency memoranda clarifying regulatory scope and application

Without direct citation to enacted regulations and official privacy assessments, analysis would be unsupported by authoritative sources.

**Authority Assessment**: Insufficient regulatory documentation
**Evidence Confidence**: 0.0 - No verified federal AI surveillance regulations accessed""",
                "authority_type": "insufficient",
                "authority_score": 0.0,
                "evidence_confidence": 0.0
            }
        ]
        
        for scenario in insufficient_evidence_scenarios:
            examples.append({
                "prompt": scenario["prompt"],
                "completion": scenario["completion"],
                "metadata": {
                    "type": "evidence_or_silence",
                    "subtype": "insufficient_evidence",
                    "authority_type": scenario["authority_type"],
                    "authority_score": scenario["authority_score"],
                    "evidence_confidence": scenario["evidence_confidence"],
                    "demonstrates_principle": "evidence_or_silence"
                }
            })
        
        return examples

    def create_high_confidence_examples(self) -> List[Dict[str, Any]]:
        """Create examples with strong evidence backing"""
        examples = []
        
        high_confidence_scenarios = [
            {
                "prompt": "What are the statutory requirements for environmental impact statements under NEPA?",
                "completion": """### Analysis:

The National Environmental Policy Act (NEPA) establishes specific statutory requirements for Environmental Impact Statements (EIS):

**Core Requirements** [42 USC § 4332(C)]:
- Federal agencies must prepare detailed statements on environmental impacts of major federal actions significantly affecting human environment
- Analysis must include environmental impact, adverse effects that cannot be avoided, and alternatives to proposed action

**Procedural Requirements** [40 CFR § 1502]:
- Public scoping process to determine scope of issues
- Draft EIS preparation and public comment period (minimum 45 days)
- Final EIS addressing comments and responses
- Record of Decision explaining agency's decision and mitigation measures

**Timing Requirements** [40 CFR § 1506.10]:
- No irreversible commitments of resources before final EIS completion
- Minimum 30-day waiting period after Final EIS publication before Record of Decision

**Authority Assessment**: Statutory and regulatory authority (confidence: 1.0)
**Evidence Confidence**: 0.95 - Direct citation to USC and CFR provisions""",
                "authority_type": "statute",
                "authority_score": 1.0,
                "evidence_confidence": 0.95
            },
            {
                "prompt": "What enforcement mechanisms exist under the Clean Air Act for violation of emissions standards?",
                "completion": """### Analysis:

The Clean Air Act provides comprehensive enforcement mechanisms for emissions standard violations:

**Civil Enforcement** [42 USC § 7413(b)]:
- EPA may issue administrative penalty orders up to $25,000 per day of violation
- Federal court civil actions for injunctive relief and civil penalties up to $37,500 per day per violation [adjusted for inflation per 40 CFR § 19]

**Criminal Enforcement** [42 USC § 7413(c)]:
- Knowing violations: fines up to $50,000 per day, imprisonment up to 2 years
- Knowing endangerment: fines up to $250,000 (individuals) or $500,000 (organizations), imprisonment up to 15 years

**Administrative Tools** [42 USC § 7413(a)]:
- Compliance orders requiring cessation of violations
- Administrative penalty orders without court proceedings
- Field citations for minor violations [42 USC § 7413(d)]

**Authority Assessment**: Direct statutory authority (confidence: 1.0)
**Evidence Confidence**: 0.92 - Comprehensive USC citation with current penalty amounts""",
                "authority_type": "statute",
                "authority_score": 1.0,
                "evidence_confidence": 0.92
            }
        ]
        
        for scenario in high_confidence_scenarios:
            examples.append({
                "prompt": scenario["prompt"],
                "completion": scenario["completion"],
                "metadata": {
                    "type": "evidence_or_silence",
                    "subtype": "high_confidence",
                    "authority_type": scenario["authority_type"],
                    "authority_score": scenario["authority_score"],
                    "evidence_confidence": scenario["evidence_confidence"],
                    "demonstrates_principle": "evidence_based_analysis"
                }
            })
        
        return examples

    def create_qualified_response_examples(self) -> List[Dict[str, Any]]:
        """Create examples with qualified responses noting evidence limitations"""
        examples = []
        
        qualified_scenarios = [
            {
                "prompt": "How do federal tax incentives for renewable energy affect deployment rates?",
                "completion": """### Analysis:

Based on available government sources, federal tax incentives demonstrate measurable impact on renewable energy deployment:

**Investment Tax Credit (ITC) Impact** [IRS Revenue Procedure 2020-12]:
- Solar installations: 30% credit through 2022, stepping down to 26% (2023), 22% (2024)
- Wind installations: Production Tax Credit of $0.025/kWh (2023 rate)

**Deployment Correlation** [EIA Electric Power Annual 2023]:
- Solar capacity grew from 40 GW (2019) to 131 GW (2023) during ITC availability
- Wind capacity increased from 96 GW (2019) to 142 GW (2023)

**Limitations**: Direct causation analysis requires econometric studies controlling for technology cost reductions, state policies, and market factors. Available EIA data shows correlation but cannot isolate tax credit impact from other variables.

**Authority Assessment**: Government statistical data (confidence: 0.8)  
**Evidence Confidence**: 0.72 - Strong correlational data, limited causal analysis""",
                "authority_type": "agency_report",
                "authority_score": 0.8,
                "evidence_confidence": 0.72
            }
        ]
        
        for scenario in qualified_scenarios:
            examples.append({
                "prompt": scenario["prompt"],
                "completion": scenario["completion"],
                "metadata": {
                    "type": "evidence_or_silence",
                    "subtype": "qualified_response",
                    "authority_type": scenario["authority_type"],
                    "authority_score": scenario["authority_score"],
                    "evidence_confidence": scenario["evidence_confidence"],
                    "demonstrates_principle": "qualified_evidence_analysis"
                }
            })
        
        return examples

    def generate_all_examples(self) -> List[Dict[str, Any]]:
        """Generate complete set of evidence-or-silence training examples"""
        logger.info("Generating evidence-or-silence training examples...")
        
        all_examples = []
        
        # Add insufficient evidence examples
        insufficient_examples = self.create_insufficient_evidence_examples()
        all_examples.extend(insufficient_examples)
        logger.info(f"Created {len(insufficient_examples)} insufficient evidence examples")
        
        # Add high confidence examples
        high_confidence_examples = self.create_high_confidence_examples()
        all_examples.extend(high_confidence_examples)
        logger.info(f"Created {len(high_confidence_examples)} high confidence examples")
        
        # Add qualified response examples
        qualified_examples = self.create_qualified_response_examples()
        all_examples.extend(qualified_examples)
        logger.info(f"Created {len(qualified_examples)} qualified response examples")
        
        logger.info(f"Total evidence-or-silence examples: {len(all_examples)}")
        return all_examples

    def save_examples_to_file(self, examples: List[Dict[str, Any]], output_file: str):
        """Save examples to JSONL file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
        
        logger.info(f"Saved {len(examples)} examples to {output_file}")

def main():
    """Main execution function"""
    logger.info("🔍 Evidence-or-Silence Examples Generator")
    
    # Initialize generator
    generator = EvidenceOrSilenceExamples()
    
    # Generate examples
    examples = generator.generate_all_examples()
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"evidence_or_silence_examples_{timestamp}.jsonl"
    generator.save_examples_to_file(examples, output_file)
    
    # Create summary
    summary = {
        "total_examples": len(examples),
        "types": {},
        "authority_distribution": {},
        "confidence_range": {
            "min": min(ex["metadata"]["evidence_confidence"] for ex in examples),
            "max": max(ex["metadata"]["evidence_confidence"] for ex in examples),
            "avg": sum(ex["metadata"]["evidence_confidence"] for ex in examples) / len(examples)
        }
    }
    
    # Count by type
    for example in examples:
        subtype = example["metadata"]["subtype"]
        summary["types"][subtype] = summary["types"].get(subtype, 0) + 1
        
        authority = example["metadata"]["authority_type"]
        summary["authority_distribution"][authority] = summary["authority_distribution"].get(authority, 0) + 1
    
    # Save summary
    summary_file = f"evidence_or_silence_summary_{timestamp}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info("📊 Generation Summary:")
    logger.info(f"  Total examples: {summary['total_examples']}")
    logger.info(f"  Types: {summary['types']}")
    logger.info(f"  Authority distribution: {summary['authority_distribution']}")
    logger.info(f"  Confidence range: {summary['confidence_range']['min']:.2f} - {summary['confidence_range']['max']:.2f}")
    logger.info(f"  Files created: {output_file}, {summary_file}")
    
    return examples

if __name__ == "__main__":
    examples = main()