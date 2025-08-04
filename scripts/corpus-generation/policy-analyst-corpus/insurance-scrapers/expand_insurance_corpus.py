#!/usr/bin/env python3
"""
Expand Insurance Corpus for Fine-tuning
Aggressive collection to reach ~500+ papers minimum for viable training
"""

import logging
from scihub_enhanced_scraper import SciHubEnhancedScraper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("corpus_expander")

class InsuranceCorpusExpander(SciHubEnhancedScraper):
    """Aggressive expansion for fine-tuning corpus"""
    
    def __init__(self, s3_bucket: str = "policy-database", s3_prefix: str = "insurance"):
        super().__init__(s3_bucket, s3_prefix)
        
        # More aggressive search terms for broader coverage
        self.extended_search_terms = [
            # Core insurance
            'insurance pricing models',
            'actuarial mathematics',
            'life insurance reserves',
            'property casualty insurance',
            'health insurance economics',
            'motor vehicle insurance',
            'marine insurance coverage',
            'aviation insurance risk',
            
            # Risk management
            'enterprise risk management ERM',
            'operational risk measurement',
            'credit risk insurance',
            'market risk insurance',
            'liquidity risk management',
            'concentration risk',
            'model risk validation',
            'stress testing methodologies',
            
            # Regulatory & compliance
            'insurance supervision',
            'capital adequacy requirements',
            'risk-based capital RBC',
            'own risk solvency assessment ORSA',
            'insurance regulation compliance',
            'prudential regulation',
            'macroprudential policy',
            
            # Specialized risks
            'catastrophe risk modeling',
            'natural disaster insurance',
            'earthquake insurance',
            'flood insurance programs',
            'terrorism risk insurance',
            'cyber security insurance',
            'pandemic business interruption',
            'climate change insurance',
            
            # Reinsurance
            'reinsurance treaty structures',
            'proportional reinsurance',
            'non-proportional reinsurance',
            'excess of loss reinsurance',
            'quota share reinsurance',
            'surplus reinsurance',
            'facultative reinsurance',
            
            # Alternative risk transfer
            'insurance linked securities ILS',
            'catastrophe bonds cat bonds',
            'insurance derivatives',
            'weather derivatives',
            'mortality bonds',
            'longevity swaps',
            'sidecars insurance',
            
            # Emerging areas
            'insurtech innovation',
            'parametric insurance',
            'microinsurance developing markets',
            'peer-to-peer insurance',
            'blockchain insurance',
            'artificial intelligence insurance',
            'telematics insurance',
            'usage-based insurance UBI',
            
            # Financial aspects
            'insurance investment strategy',
            'asset liability management ALM',
            'fair value accounting insurance',
            'embedded value',
            'economic capital',
            'IFRS 17 insurance contracts',
            
            # Behavioral & social
            'insurance consumer behavior',
            'moral hazard insurance',
            'adverse selection',
            'insurance fraud detection',
            'social insurance systems',
            'pension fund management'
        ]
    
    def expand_arxiv_comprehensively(self) -> list:
        """Comprehensive arXiv search with all terms"""
        all_papers = []
        
        for i, term in enumerate(self.extended_search_terms):
            logger.info(f"ArXiv search {i+1}/{len(self.extended_search_terms)}: {term}")
            papers = self.search_arxiv_papers(term, max_results=10)
            all_papers.extend(papers)
            
            # Don't overwhelm arXiv
            if i % 10 == 0:
                time.sleep(5)
        
        return all_papers
    
    def expand_semantic_scholar_comprehensively(self) -> list:
        """Comprehensive Semantic Scholar search"""
        all_papers = []
        
        # Focus on high-impact terms for Semantic Scholar
        high_impact_terms = [
            term for term in self.extended_search_terms 
            if any(keyword in term.lower() for keyword in [
                'solvency', 'capital', 'regulation', 'catastrophe', 
                'reinsurance', 'cyber', 'climate', 'pandemic'
            ])
        ]
        
        for i, term in enumerate(high_impact_terms):
            logger.info(f"Semantic Scholar search {i+1}/{len(high_impact_terms)}: {term}")
            papers = self.search_semantic_scholar(term, max_results=15)
            all_papers.extend(papers)
            
            # Rate limiting
            time.sleep(2)
        
        return all_papers
    
    def search_additional_repositories(self) -> list:
        """Search additional academic repositories"""
        papers = []
        
        # More institutional sources with working URLs
        working_sources = [
            {
                'title': 'BIS Working Papers on Insurance',
                'url': 'https://www.bis.org/publ/work890.pdf',
                'source': 'BIS'
            },
            {
                'title': 'ECB Financial Stability Review',
                'url': 'https://www.ecb.europa.eu/pub/pdf/fsr/ecb.fsr202305~cd7eeb5cca.en.pdf',
                'source': 'ECB'
            },
            {
                'title': 'Fed Supervision Manual',
                'url': 'https://www.federalreserve.gov/publications/files/svb-supervision-sr-letters-23-4.pdf',
                'source': 'Federal Reserve'
            }
        ]
        
        for doc in working_sources:
            try:
                response = self.session.get(doc['url'], timeout=30)
                if response.status_code == 200:
                    paper = {
                        'title': doc['title'],
                        'pdf_url': doc['url'],
                        'source': doc['source'],
                        'content': response.content
                    }
                    papers.append(paper)
                    logger.info(f"✓ Downloaded: {doc['title']}")
                time.sleep(2)
            except Exception as e:
                logger.warning(f"Failed to download {doc['title']}: {str(e)}")
        
        return papers
    
    def aggressive_expansion(self, target_minimum: int = 300) -> list:
        """Aggressively expand to minimum viable corpus size"""
        logger.info(f"🚀 Starting aggressive expansion to {target_minimum} papers minimum")
        
        current_count = len(self.downloaded_papers)
        logger.info(f"Current papers: {current_count}")
        
        all_papers = []
        
        # Phase 1: Comprehensive arXiv
        logger.info("Phase 1: Comprehensive arXiv search...")
        arxiv_papers = self.expand_arxiv_comprehensively()
        all_papers.extend(arxiv_papers)
        logger.info(f"Found {len(arxiv_papers)} arXiv papers")
        
        # Phase 2: Comprehensive Semantic Scholar  
        logger.info("Phase 2: Comprehensive Semantic Scholar search...")
        semantic_papers = self.expand_semantic_scholar_comprehensively()
        all_papers.extend(semantic_papers)
        logger.info(f"Found {len(semantic_papers)} Semantic Scholar papers")
        
        # Phase 3: Additional repositories
        logger.info("Phase 3: Additional institutional sources...")
        repo_papers = self.search_additional_repositories()
        all_papers.extend(repo_papers)
        logger.info(f"Found {len(repo_papers)} institutional papers")
        
        # Deduplicate and rank
        unique_papers = self._deduplicate_papers(all_papers)
        ranked_papers = self._rank_papers_by_quality(unique_papers)
        
        logger.info(f"After deduplication: {len(ranked_papers)} unique papers")
        
        # Download with Sci-Hub fallback
        needed = target_minimum - current_count
        logger.info(f"Need {needed} more papers to reach {target_minimum}")
        
        successful_uploads = 0
        scihub_successes = 0
        
        for i, paper in enumerate(ranked_papers):
            if current_count + successful_uploads >= target_minimum:
                break
                
            logger.info(f"Processing {i+1}/{len(ranked_papers)}: {paper['title'][:50]}...")
            
            # Download with Sci-Hub fallback
            pdf_content = None
            used_scihub = False
            
            if 'content' in paper:
                pdf_content = paper['content']
            else:
                pdf_content = self.download_pdf(paper['pdf_url'], paper['title'])
                
                if not pdf_content:
                    pdf_content = self.download_from_scihub_for_paper(paper)
                    if pdf_content:
                        used_scihub = True
                        scihub_successes += 1
            
            if not pdf_content:
                continue
            
            # Process and upload
            pdf_metadata = self.extract_pdf_metadata(pdf_content)
            
            combined_metadata = {
                'title': paper.get('title', ''),
                'source': paper.get('source', ''),
                'pdf_url': paper.get('pdf_url', ''),
                'collection_date': datetime.now().isoformat(),
                'scihub_used': used_scihub,
                'expansion_phase': True,
                **pdf_metadata
            }
            
            clean_metadata = self._clean_metadata_for_s3(combined_metadata)
            filename = self.generate_filename(paper)
            
            if self.upload_to_s3(pdf_content, filename, clean_metadata):
                successful_uploads += 1
                self.downloaded_papers.append({
                    'filename': filename,
                    's3_key': f"{self.s3_prefix}/{filename}",
                    'metadata': clean_metadata
                })
                
                total = current_count + successful_uploads
                source_note = " [Sci-Hub]" if used_scihub else ""
                logger.info(f"✓ Uploaded {total}/{target_minimum}: {filename}{source_note}")
            
            time.sleep(1)
        
        final_count = current_count + successful_uploads
        logger.info(f"🎉 Expansion complete!")
        logger.info(f"📊 Total papers: {final_count}")
        logger.info(f"🔬 Additional Sci-Hub successes: {scihub_successes}")
        logger.info(f"{'✅ Viable for fine-tuning!' if final_count >= 300 else '⚠️ Still below recommended 300+ for robust training'}")
        
        return self.downloaded_papers

def main():
    """Run aggressive expansion"""
    expander = InsuranceCorpusExpander()
    
    # Expand to minimum viable corpus size
    papers = expander.aggressive_expansion(target_minimum=300)
    
    # Save expanded catalog
    expander.save_catalog("insurance_expanded_corpus_catalog.json")
    
    print(f"📈 Expansion complete! {len(papers)} papers total")
    if len(papers) >= 300:
        print("✅ Ready for fine-tuning!")
    else:
        print(f"⚠️ Recommend expanding further (current: {len(papers)}, target: 300+)")

if __name__ == "__main__":
    main()