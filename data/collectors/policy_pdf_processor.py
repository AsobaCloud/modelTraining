#!/usr/bin/env python3
"""
Policy PDF Text Processor
Extract text from policy PDFs and convert to training-ready format
Following CLAUDE.md real-world data collection principles
"""

import os
import re
import json
import boto3
import logging
import hashlib
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import PyPDF2
    import pdfplumber
except ImportError:
    print("Installing required PDF processing packages...")
    os.system("pip install PyPDF2 pdfplumber")
    import PyPDF2
    import pdfplumber

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class PolicyPDFProcessor:
    """Process PDFs and extract text for policy analysis training"""
    
    def __init__(self, bucket_name: str = "policy-database", region: str = "us-east-1"):
        """Initialize processor with S3 configuration"""
        self.bucket_name = bucket_name
        self.region = region
        self.s3_client = boto3.client('s3', region_name=region)
        
        # Processing statistics
        self.stats = {
            "pdfs_processed": 0,
            "text_extraction_successes": 0,
            "text_extraction_failures": 0,
            "training_examples_generated": 0,
            "total_characters_extracted": 0,
            "average_document_length": 0,
            "quality_filtered": 0
        }
        
        # Policy domain classifications
        self.policy_domains = {
            "economic": ["economic", "fiscal", "monetary", "budget", "tax", "trade", "employment", "gdp", "inflation"],
            "energy": ["energy", "electricity", "renewable", "carbon", "oil", "gas", "solar", "wind", "nuclear"],
            "regulatory": ["regulation", "compliance", "enforcement", "rules", "standards", "oversight", "agency"],
            "legislative": ["bill", "act", "legislation", "congress", "house", "senate", "amendment", "law"],
            "judicial": ["court", "ruling", "decision", "supreme", "federal", "circuit", "justice", "legal"],
            "international": ["international", "foreign", "treaty", "trade", "diplomatic", "embassy", "global"]
        }

    def download_pdf(self, s3_key: str) -> Optional[str]:
        """Download PDF from S3 to temporary file"""
        try:
            # Create temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.pdf')
            os.close(temp_fd)
            
            # Download file
            logger.debug(f"Downloading {s3_key}")
            self.s3_client.download_file(self.bucket_name, s3_key, temp_path)
            
            # Verify file size
            if os.path.getsize(temp_path) > 0:
                return temp_path
            else:
                logger.warning(f"Downloaded PDF is empty: {s3_key}")
                os.remove(temp_path)
                return None
                
        except Exception as e:
            logger.error(f"Error downloading {s3_key}: {e}")
            return None

    def extract_text_pdfplumber(self, pdf_path: str) -> Optional[str]:
        """Extract text using pdfplumber (more reliable)"""
        try:
            text_content = []
            
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text_content.append(page_text.strip())
                    except Exception as e:
                        logger.debug(f"Error extracting page {page_num} from {pdf_path}: {e}")
                        continue
            
            if text_content:
                full_text = "\n\n".join(text_content)
                return full_text
            else:
                return None
                
        except Exception as e:
            logger.debug(f"pdfplumber extraction failed for {pdf_path}: {e}")
            return None

    def extract_text_pypdf2(self, pdf_path: str) -> Optional[str]:
        """Extract text using PyPDF2 (fallback method)"""
        try:
            text_content = []
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                for page_num in range(len(pdf_reader.pages)):
                    try:
                        page = pdf_reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            text_content.append(page_text.strip())
                    except Exception as e:
                        logger.debug(f"Error extracting page {page_num} from {pdf_path}: {e}")
                        continue
            
            if text_content:
                full_text = "\n\n".join(text_content)
                return full_text
            else:
                return None
                
        except Exception as e:
            logger.debug(f"PyPDF2 extraction failed for {pdf_path}: {e}")
            return None

    def extract_pdf_text(self, pdf_path: str) -> Optional[str]:
        """Extract text from PDF using multiple methods"""
        # Try pdfplumber first (more reliable)
        text = self.extract_text_pdfplumber(pdf_path)
        
        # Fallback to PyPDF2 if pdfplumber fails
        if not text:
            text = self.extract_text_pypdf2(pdf_path)
        
        return text

    def clean_extracted_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common PDF artifacts
        text = re.sub(r'(?:\s*\n\s*){3,}', '\n\n', text)  # Excessive line breaks
        text = re.sub(r'\s*\.\s*\.\s*\.+', '...', text)  # Multiple dots
        text = re.sub(r'[^\w\s\.,;:!?\-\(\)\[\]\'\"]+', ' ', text)  # Non-standard characters
        
        # Remove headers/footers patterns
        text = re.sub(r'^\s*page\s+\d+\s*$', '', text, flags=re.IGNORECASE | re.MULTILINE)
        text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
        
        # Normalize spacing
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def classify_policy_domain(self, text: str) -> List[str]:
        """Classify the policy domain(s) of the document"""
        text_lower = text.lower()
        domains = []
        
        for domain, keywords in self.policy_domains.items():
            keyword_matches = sum(1 for keyword in keywords if keyword in text_lower)
            if keyword_matches >= 2:  # Require at least 2 keyword matches
                domains.append(domain)
        
        return domains if domains else ["general"]

    def classify_document_authority(self, text: str, source_info: Dict[str, Any]) -> Tuple[str, float]:
        """Classify document authority level using evidence-first hierarchy"""
        text_lower = text.lower()
        source_key = source_info.get("key", "").lower()
        
        # Authority cascade: law > regulation > agency memo > peer-review > news/blog
        authority_hierarchy = {
            "statute": 1.0,
            "regulation": 0.9,
            "executive_order": 0.9,
            "court_decision": 0.9,
            "agency_report": 0.8,
            "congressional_report": 0.8,
            "agency_memo": 0.7,
            "government_document": 0.6,
            "unknown": 0.3
        }
        
        # Check for statutory language (highest authority)
        if any(pattern in text_lower for pattern in ["usc", "united states code", "section", "§", "statute"]):
            return "statute", authority_hierarchy["statute"]
        
        # Check for regulatory language
        if any(pattern in text_lower for pattern in ["cfr", "code of federal regulations", "regulation"]):
            return "regulation", authority_hierarchy["regulation"]
            
        # Check for executive orders
        if any(pattern in text_lower for pattern in ["executive order", "presidential directive"]):
            return "executive_order", authority_hierarchy["executive_order"]
            
        # Check for court decisions
        if any(pattern in text_lower for pattern in ["court", "circuit", "district", "supreme court", "ruling", "judgment"]):
            return "court_decision", authority_hierarchy["court_decision"]
            
        # Check for congressional materials
        if any(pattern in text_lower for pattern in ["congress", "house", "senate", "committee report"]):
            return "congressional_report", authority_hierarchy["congressional_report"]
            
        # Check for agency reports
        if any(pattern in text_lower for pattern in ["agency report", "department of", "administration", "commission"]):
            return "agency_report", authority_hierarchy["agency_report"]
            
        # Check source folder for additional context
        if "federal-legislation" in source_key:
            return "statute", authority_hierarchy["statute"]
        elif "congressional-research" in source_key:
            return "congressional_report", authority_hierarchy["congressional_report"]
        elif "exec_orders" in source_key:
            return "executive_order", authority_hierarchy["executive_order"]
        elif "energy-trading" in source_key:
            return "regulation", authority_hierarchy["regulation"]
            
        return "government_document", authority_hierarchy["government_document"]

    def extract_document_metadata(self, text: str, source_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from document text and source with authority cascade"""
        # Classify document authority using evidence-first hierarchy
        authority_type, authority_score = self.classify_document_authority(text, source_info)
        
        metadata = {
            "source": source_info,
            "extraction_date": datetime.now().isoformat(),
            "character_count": len(text),
            "word_count": len(text.split()),
            "domains": self.classify_policy_domain(text),
            "authority_type": authority_type,
            "authority_score": authority_score,
            "quality_score": 0.0,
            "evidence_confidence": 0.0
        }
        
        # Calculate quality score based on various factors
        quality_factors = []
        
        # Length factor (prefer medium-length documents)
        char_count = len(text)
        if 1000 <= char_count <= 50000:
            quality_factors.append(1.0)
        elif char_count < 1000:
            quality_factors.append(char_count / 1000.0)
        else:
            quality_factors.append(50000.0 / char_count)
        
        # Content structure factor
        has_structure = any(pattern in text for pattern in [
            "introduction", "background", "analysis", "recommendation", 
            "conclusion", "summary", "overview", "policy", "section"
        ])
        quality_factors.append(1.0 if has_structure else 0.5)
        
        # Domain relevance factor
        domain_score = min(len(metadata["domains"]) / 2.0, 1.0)  # Up to 2 domains is ideal
        quality_factors.append(domain_score)
        
        # Authority factor - higher authority documents get quality boost
        quality_factors.append(authority_score)
        
        # Calculate average quality score
        base_quality = sum(quality_factors) / len(quality_factors)
        
        # Evidence confidence scoring
        evidence_factors = []
        
        # Authority level directly impacts evidence confidence
        evidence_factors.append(authority_score)
        
        # Check for factual precision indicators
        has_citations = any(pattern in text for pattern in ["§", "USC", "CFR", "see", "pursuant to", "under"])
        evidence_factors.append(0.8 if has_citations else 0.3)
        
        # Check for quantitative data
        has_data = bool(re.search(r'\d+%|\d+\.\d+%|\$\d+|\d+\s+(million|billion|trillion)', text))
        evidence_factors.append(0.7 if has_data else 0.4)
        
        # Check for hedging language (reduces confidence)
        hedging_words = ["may", "might", "could", "possibly", "potentially", "appears", "seems"]
        hedge_count = sum(1 for word in hedging_words if word in text.lower())
        hedge_penalty = min(0.3, hedge_count * 0.05)
        
        # Calculate evidence confidence
        evidence_confidence = (sum(evidence_factors) / len(evidence_factors)) - hedge_penalty
        evidence_confidence = max(0.1, min(1.0, evidence_confidence))
        
        # Final quality score incorporates evidence confidence
        metadata["quality_score"] = (base_quality * 0.7) + (evidence_confidence * 0.3)
        metadata["evidence_confidence"] = evidence_confidence
        
        return metadata

    def generate_training_examples(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate training examples from document text"""
        examples = []
        
        # Split text into manageable chunks
        max_chunk_size = 2000
        chunks = self.split_text_into_chunks(text, max_chunk_size)
        
        # Generate different types of analysis prompts
        for i, chunk in enumerate(chunks[:3]):  # Limit to first 3 chunks per document
            if len(chunk.strip()) < 200:  # Skip very short chunks
                continue
            
            # Create base example with document summary
            if i == 0:  # First chunk - document summary
                examples.append({
                    "prompt": f"Provide a policy analysis summary of this document from the {', '.join(metadata['domains'])} domain(s).",
                    "completion": self.create_summary_response(chunk, metadata),
                    "metadata": {
                        "type": "summary",
                        "domains": metadata["domains"],
                        "source": metadata["source"],
                        "chunk_index": i,
                        "quality_score": metadata["quality_score"],
                        "authority_type": metadata["authority_type"],
                        "authority_score": metadata["authority_score"],
                        "evidence_confidence": metadata["evidence_confidence"]
                    }
                })
            
            # Create domain-specific analysis examples
            for domain in metadata["domains"]:
                if domain != "general":
                    examples.append({
                        "prompt": f"Analyze the {domain} implications of the following policy content:",
                        "completion": self.create_domain_analysis_response(chunk, domain, metadata),
                        "metadata": {
                            "type": "domain_analysis",
                            "domain": domain,
                            "source": metadata["source"],
                            "chunk_index": i,
                            "quality_score": metadata["quality_score"],
                            "authority_type": metadata["authority_type"],
                            "authority_score": metadata["authority_score"],
                            "evidence_confidence": metadata["evidence_confidence"]
                        }
                    })
        
        # Generate policy recommendations example
        if len(text) > 1000:  # Only for substantial documents
            examples.append({
                "prompt": "What are the key policy recommendations that emerge from this analysis?",
                "completion": self.create_recommendations_response(text, metadata),
                "metadata": {
                    "type": "recommendations",
                    "domains": metadata["domains"],
                    "source": metadata["source"],
                    "quality_score": metadata["quality_score"],
                    "authority_type": metadata["authority_type"],
                    "authority_score": metadata["authority_score"],
                    "evidence_confidence": metadata["evidence_confidence"]
                }
            })
        
        return examples

    def split_text_into_chunks(self, text: str, max_size: int) -> List[str]:
        """Split text into chunks while preserving sentence boundaries"""
        if len(text) <= max_size:
            return [text]
        
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence_size = len(sentence)
            
            if current_size + sentence_size > max_size and current_chunk:
                # Finish current chunk
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
        
        # Add final chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks

    def create_summary_response(self, text: str, metadata: Dict[str, Any]) -> str:
        """Create a policy analysis summary response"""
        domains = metadata["domains"]
        domain_str = f"{', '.join(domains)} policy" if domains != ["general"] else "policy"
        
        # Extract key points from text
        text_preview = text[:800] + "..." if len(text) > 800 else text
        
        return f"""## {domain_str.title()} Analysis Summary

### Overview
This document addresses key aspects of {domain_str} with implications for policy development and implementation.

### Key Content
{text_preview}

### Policy Significance
This analysis provides insights into {domain_str} considerations that inform evidence-based policy decisions. The document contributes to understanding of regulatory frameworks and their practical applications.

### Recommendations for Further Analysis
- Review implementation mechanisms and enforcement strategies
- Consider stakeholder impact assessments  
- Evaluate alignment with existing policy frameworks
- Assess resource requirements and feasibility constraints"""

    def create_domain_analysis_response(self, text: str, domain: str, metadata: Dict[str, Any]) -> str:
        """Create domain-specific analysis response"""
        text_preview = text[:600] + "..." if len(text) > 600 else text
        
        domain_context = {
            "economic": "economic impact, fiscal implications, and market effects",
            "energy": "energy policy, resource management, and sustainability considerations",
            "regulatory": "regulatory compliance, enforcement mechanisms, and oversight structures",
            "legislative": "legislative intent, legal frameworks, and statutory requirements",
            "judicial": "legal precedent, court decisions, and judicial interpretation",
            "international": "international relations, diplomatic considerations, and global implications"
        }
        
        context = domain_context.get(domain, f"{domain} policy implications")
        
        return f"""## {domain.title()} Policy Analysis

### {domain.title()} Context
This analysis examines the {context} within the following policy content:

{text_preview}

### {domain.title()} Implications
The policy framework presented has significant implications for {context}. Key considerations include:

- Strategic alignment with {domain} objectives
- Implementation challenges and opportunities
- Stakeholder impact assessment
- Resource allocation and operational requirements

### {domain.title()} Recommendations
Based on this analysis, policy makers should consider the {domain} dimensions when developing implementation strategies and evaluating policy effectiveness."""

    def create_recommendations_response(self, text: str, metadata: Dict[str, Any]) -> str:
        """Create policy recommendations response"""
        domains = metadata["domains"]
        
        return f"""## Policy Recommendations

### Strategic Recommendations
Based on the comprehensive analysis of this {', '.join(domains)} policy document, the following recommendations emerge:

#### Implementation Strategy
1. **Phased Approach**: Implement policy changes in structured phases to ensure effective transition
2. **Stakeholder Engagement**: Conduct thorough consultation with affected parties
3. **Resource Planning**: Allocate adequate resources for implementation and monitoring

#### Operational Considerations  
1. **Monitoring Framework**: Establish metrics and evaluation criteria
2. **Compliance Mechanisms**: Develop clear enforcement and oversight procedures
3. **Risk Management**: Identify and mitigate potential implementation risks

#### Long-term Sustainability
1. **Regular Review**: Schedule periodic policy effectiveness assessments  
2. **Adaptive Management**: Build flexibility for policy adjustments
3. **Capacity Building**: Invest in institutional capacity for ongoing implementation

These recommendations provide a framework for translating policy analysis into actionable implementation strategies."""

    def process_pdf_file(self, pdf_info: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single PDF file"""
        result = {
            "pdf_info": pdf_info,
            "success": False,
            "text_extracted": False,
            "training_examples": [],
            "processing_time": 0,
            "error": None
        }
        
        start_time = datetime.now()
        
        try:
            # Download PDF
            if "s3_key" in pdf_info:
                # S3 PDF
                pdf_path = self.download_pdf(pdf_info["s3_key"])
                source_info = {"type": "s3", "key": pdf_info["s3_key"]}
            else:
                # Local extracted PDF
                pdf_path = pdf_info["local_path"]
                source_info = {"type": "extracted", "source_zip": pdf_info.get("source_zip", "unknown")}
            
            if not pdf_path:
                result["error"] = "Failed to download/access PDF"
                return result
            
            try:
                # Extract text
                logger.debug(f"Extracting text from {pdf_path}")
                raw_text = self.extract_pdf_text(pdf_path)
                
                if raw_text:
                    # Clean text
                    cleaned_text = self.clean_extracted_text(raw_text)
                    
                    if len(cleaned_text) > 100:  # Minimum viable content
                        result["text_extracted"] = True
                        self.stats["text_extraction_successes"] += 1
                        self.stats["total_characters_extracted"] += len(cleaned_text)
                        
                        # Extract metadata with authority and confidence scoring
                        metadata = self.extract_document_metadata(cleaned_text, source_info)
                        
                        # Quality filter with authority consideration
                        min_quality = 0.3
                        if metadata["authority_type"] in ["statute", "regulation", "court_decision"]:
                            min_quality = 0.2  # Lower threshold for high-authority documents
                        
                        if metadata["quality_score"] >= min_quality:
                            # Generate training examples with confidence metadata
                            training_examples = self.generate_training_examples(cleaned_text, metadata)
                            
                            result["training_examples"] = training_examples
                            result["success"] = True
                            result["authority_info"] = {
                                "type": metadata["authority_type"],
                                "score": metadata["authority_score"],
                                "evidence_confidence": metadata["evidence_confidence"]
                            }
                            
                            self.stats["training_examples_generated"] += len(training_examples)
                            
                            logger.debug(f"Generated {len(training_examples)} training examples from {pdf_path} "
                                       f"(authority: {metadata['authority_type']}, confidence: {metadata['evidence_confidence']:.2f})")
                        else:
                            result["error"] = f"Quality score too low: {metadata['quality_score']:.2f} (min: {min_quality})"
                            self.stats["quality_filtered"] += 1
                    else:
                        result["error"] = f"Insufficient text content: {len(cleaned_text)} characters"
                        self.stats["text_extraction_failures"] += 1
                else:
                    result["error"] = "No text could be extracted from PDF"
                    self.stats["text_extraction_failures"] += 1
            
            finally:
                # Clean up downloaded file
                if "s3_key" in pdf_info and pdf_path and os.path.exists(pdf_path):
                    try:
                        os.remove(pdf_path)
                    except:
                        pass
            
            self.stats["pdfs_processed"] += 1
            
        except Exception as e:
            result["error"] = f"Processing error: {str(e)}"
            logger.error(f"Error processing PDF {pdf_info}: {e}")
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        result["processing_time"] = processing_time
        
        return result

    def process_pdf_collection(self, collection_file: str, max_pdfs: int = 1000) -> Dict[str, Any]:
        """Process PDFs from collection results file"""
        logger.info(f"🚀 Starting PDF processing from {collection_file}")
        
        # Load collection results
        try:
            with open(collection_file, 'r', encoding='utf-8') as f:
                collection_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading collection file {collection_file}: {e}")
            return {"error": f"Failed to load collection file: {e}"}
        
        processing_results = {
            "start_time": datetime.now().isoformat(),
            "collection_source": collection_file,
            "processed_pdfs": [],
            "all_training_examples": [],
            "processing_stats": {},
            "errors": []
        }
        
        # Collect all PDFs from collection results
        all_pdfs = []
        
        for folder, folder_data in collection_data.get("folders", {}).items():
            # Add direct PDFs
            for pdf_info in folder_data.get("pdf_files", []):
                pdf_info["folder"] = folder
                all_pdfs.append(pdf_info)
            
            # Add extracted PDFs
            for pdf_info in folder_data.get("extracted_pdfs", []):
                pdf_info["folder"] = folder
                all_pdfs.append(pdf_info)
        
        logger.info(f"Found {len(all_pdfs)} PDFs to process")
        
        # Limit processing if specified
        if max_pdfs and len(all_pdfs) > max_pdfs:
            all_pdfs = all_pdfs[:max_pdfs]
            logger.info(f"Limited to {max_pdfs} PDFs for processing")
        
        # Process PDFs with threading
        successful_processing = 0
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit processing tasks
            future_to_pdf = {
                executor.submit(self.process_pdf_file, pdf_info): pdf_info 
                for pdf_info in all_pdfs
            }
            
            # Collect results
            for i, future in enumerate(as_completed(future_to_pdf), 1):
                try:
                    result = future.result()
                    processing_results["processed_pdfs"].append(result)
                    
                    if result["success"]:
                        successful_processing += 1
                        processing_results["all_training_examples"].extend(result["training_examples"])
                    
                    # Progress logging
                    if i % 50 == 0 or i == len(all_pdfs):
                        logger.info(f"Processed {i}/{len(all_pdfs)} PDFs ({successful_processing} successful)")
                
                except Exception as e:
                    error_msg = f"Error processing PDF: {e}"
                    processing_results["errors"].append(error_msg)
                    logger.error(error_msg)
        
        # Calculate final statistics
        processing_results["processing_stats"] = {
            "total_pdfs_attempted": len(all_pdfs),
            "successful_processing": successful_processing,
            "success_rate": successful_processing / len(all_pdfs) if all_pdfs else 0,
            "total_training_examples": len(processing_results["all_training_examples"]),
            "extraction_stats": self.stats
        }
        
        processing_results["end_time"] = datetime.now().isoformat()
        
        logger.info("📊 Processing Summary:")
        logger.info(f"  PDFs processed: {successful_processing}/{len(all_pdfs)}")
        logger.info(f"  Success rate: {processing_results['processing_stats']['success_rate']:.1%}")
        logger.info(f"  Training examples: {len(processing_results['all_training_examples'])}")
        logger.info(f"  Text extraction successes: {self.stats['text_extraction_successes']}")
        
        return processing_results

    def save_processing_results(self, results: Dict[str, Any], output_path: str):
        """Save processing results to JSON file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"💾 Processing results saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving results to {output_path}: {e}")


def main():
    """Main execution function"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python policy_pdf_processor.py <collection_results.json> [max_pdfs]")
        sys.exit(1)
    
    collection_file = sys.argv[1]
    max_pdfs = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    
    logger.info(f"🚀 Policy PDF Processing Started")
    logger.info(f"Collection file: {collection_file}")
    logger.info(f"Max PDFs: {max_pdfs}")
    
    # Initialize processor
    processor = PolicyPDFProcessor()
    
    # Process PDFs
    results = processor.process_pdf_collection(collection_file, max_pdfs)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"policy_pdf_processing_{timestamp}.json"
    processor.save_processing_results(results, output_file)
    
    # Print final summary
    if "processing_stats" in results:
        stats = results["processing_stats"]
        logger.info("🎯 Final Processing Summary:")
        logger.info(f"  📄 PDFs processed: {stats['successful_processing']}/{stats['total_pdfs_attempted']}")
        logger.info(f"  ✅ Success rate: {stats['success_rate']:.1%}")
        logger.info(f"  📚 Training examples generated: {stats['total_training_examples']}")
        logger.info(f"  💾 Results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    results = main()