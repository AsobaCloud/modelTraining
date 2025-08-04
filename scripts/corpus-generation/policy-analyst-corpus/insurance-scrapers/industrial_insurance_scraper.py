#!/usr/bin/env python3
"""
Industrial Insurance Corpus Scraper
Target: 1000+ papers for serious fine-tuning
Multi-source, parallel collection with aggressive Sci-Hub usage
"""

import asyncio
import aiohttp
import logging
import time
import json
from concurrent.futures import ThreadPoolExecutor
from scihub_enhanced_scraper import SciHubEnhancedScraper
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("industrial_scraper")

class IndustrialInsuranceScraper(SciHubEnhancedScraper):
    """Industrial-scale scraper for 1000+ insurance papers"""
    
    def __init__(self, s3_bucket: str = "policy-database", s3_prefix: str = "insurance"):
        super().__init__(s3_bucket, s3_prefix)
        
        # Massive search term database - 200+ terms
        self.massive_search_terms = [
            # Core Insurance Fundamentals
            'insurance risk assessment', 'actuarial science methods', 'life insurance mathematics',
            'property casualty insurance', 'health insurance economics', 'disability insurance',
            'motor vehicle insurance', 'homeowners insurance', 'commercial insurance',
            'marine insurance law', 'aviation insurance coverage', 'travel insurance risk',
            'pet insurance models', 'crop insurance programs', 'livestock insurance',
            
            # Risk Management Theory
            'enterprise risk management', 'operational risk measurement', 'credit risk insurance',
            'market risk management', 'liquidity risk analysis', 'concentration risk limits',
            'model risk validation', 'stress testing methodologies', 'scenario analysis',
            'risk appetite frameworks', 'risk governance structures', 'risk culture',
            'risk reporting systems', 'key risk indicators', 'risk tolerance levels',
            
            # Regulatory Frameworks
            'solvency capital requirements', 'insurance supervision', 'prudential regulation',
            'macroprudential policy', 'systemic risk regulation', 'insurance regulation',
            'risk-based capital RBC', 'own risk solvency assessment', 'ORSA implementation',
            'regulatory capital ratios', 'minimum capital requirements', 'capital buffers',
            'supervisory review process', 'pillar 2 requirements', 'pillar 3 disclosures',
            
            # Catastrophe & Natural Disasters
            'catastrophe risk modeling', 'natural disaster insurance', 'earthquake insurance',
            'hurricane catastrophe bonds', 'flood insurance NFIP', 'wildfire insurance',
            'tornado insurance coverage', 'tsunami risk assessment', 'volcanic eruption',
            'drought insurance parametric', 'extreme weather events', 'climate catastrophes',
            'aggregate catastrophe covers', 'industry loss warranties', 'cat modeling',
            
            # Emerging Risks
            'cyber security insurance', 'cyber risk quantification', 'data breach insurance',
            'ransomware insurance coverage', 'cyber liability policies', 'cyber aggregation',
            'pandemic risk insurance', 'business interruption pandemic', 'epidemic modeling',
            'terrorism risk insurance', 'political risk insurance', 'war exclusions',
            'space insurance coverage', 'satellite insurance', 'drone insurance liability',
            
            # Climate Change & ESG
            'climate change insurance', 'climate risk disclosure', 'ESG insurance factors',
            'green insurance products', 'sustainable insurance principles', 'carbon credits',
            'environmental liability insurance', 'pollution liability coverage', 'climate adaptation',
            'transition risk insurance', 'physical climate risks', 'stranded assets',
            'nature-based solutions', 'biodiversity insurance', 'ecosystem services',
            
            # Reinsurance Structures
            'reinsurance treaty structures', 'proportional reinsurance quota', 'surplus treaties',
            'non-proportional reinsurance', 'excess of loss covers', 'stop loss reinsurance',
            'facultative reinsurance', 'automatic reinsurance', 'retrocession markets',
            'reinsurance pricing models', 'reinsurance optimization', 'reinsurance cycles',
            'alternative reinsurance', 'insurance linked securities', 'retro markets',
            
            # Alternative Risk Transfer
            'catastrophe bonds pricing', 'insurance derivatives', 'weather derivatives',
            'mortality catastrophe bonds', 'longevity risk transfer', 'longevity swaps',
            'pension risk transfer', 'sidecars reinsurance', 'collateralized reinsurance',
            'insurance securitization', 'risk transfer mechanisms', 'capital markets',
            'hybrid risk products', 'contingent capital', 'parametric triggers',
            
            # Technology & Innovation
            'insurtech digital transformation', 'artificial intelligence insurance', 'machine learning',
            'telematics insurance', 'usage-based insurance', 'pay-per-mile insurance',
            'IoT insurance applications', 'blockchain insurance', 'smart contracts',
            'parametric insurance products', 'microinsurance technology', 'peer-to-peer insurance',
            'on-demand insurance', 'digital claims processing', 'automated underwriting',
            
            # Financial & Accounting
            'fair value accounting', 'IFRS 17 implementation', 'embedded value calculation',
            'economic capital modeling', 'asset liability management', 'ALM optimization',
            'investment strategy insurance', 'liability driven investment', 'matching adjustment',
            'volatility adjustment', 'transitional measures', 'contract boundaries',
            'onerous contracts', 'contractual service margin', 'risk adjustment calculation',
            
            # Behavioral & Social Aspects
            'insurance consumer behavior', 'behavioral economics insurance', 'moral hazard',
            'adverse selection models', 'insurance fraud detection', 'claims fraud prevention',
            'underwriting fraud', 'social insurance systems', 'universal health coverage',
            'insurance inclusion gaps', 'insurance penetration', 'insurance density metrics',
            'financial inclusion insurance', 'insurance education', 'insurance literacy',
            
            # Specialized Lines
            'directors officers insurance', 'professional liability', 'medical malpractice',
            'product liability insurance', 'employment practices liability', 'fiduciary liability',
            'errors omissions insurance', 'general liability coverage', 'workers compensation',
            'surety bonds insurance', 'fidelity bonds', 'performance bonds',
            'warranty insurance products', 'credit insurance trade', 'mortgage insurance',
            
            # Life & Health Specialties
            'life insurance reserves', 'mortality table construction', 'longevity modeling',
            'health insurance underwriting', 'disability income insurance', 'long-term care',
            'critical illness insurance', 'annuity products design', 'variable annuities',
            'universal life insurance', 'whole life insurance', 'term life insurance',
            'group life insurance', 'pension fund management', 'retirement planning',
            
            # Quantitative Methods
            'extreme value theory', 'copula models insurance', 'stochastic reserving',
            'chain ladder methods', 'Bornhuetter Ferguson', 'cape cod method',
            'loss development factors', 'tail factor estimation', 'IBNR reserves',
            'claims reserving methods', 'run-off triangles', 'bootstrap methods',
            'monte carlo simulation', 'stochastic modeling', 'machine learning pricing',
            
            # International & Emerging Markets
            'microinsurance developing countries', 'index-based insurance', 'agricultural insurance',
            'crop yield insurance', 'weather index insurance', 'livestock mortality',
            'catastrophe insurance pools', 'mutual insurance societies', 'cooperative insurance',
            'takaful Islamic insurance', 'sharia compliant insurance', 'emerging markets',
            'insurance market development', 'regulatory harmonization', 'cross-border insurance'
        ]
        
        # Multiple API sources for parallel collection
        self.api_sources = {
            'crossref': 'https://api.crossref.org/works',
            'core': 'https://core.ac.uk/api-v2/articles/search',
            'openalex': 'https://api.openalex.org/works',
            'unpaywall': 'https://api.unpaywall.org/v2',
            'doaj': 'https://doaj.org/api/v2/search/articles'
        }
        
        # Working institutional URLs (verified)
        self.institutional_pdfs = [
            # Central Banks & Regulators
            'https://www.ecb.europa.eu/pub/pdf/scpwps/ecb.wp2789~8f2c3c7c3a.en.pdf',
            'https://www.bankofengland.co.uk/prudential-regulation/publication/2015/solvency-2-matching-adjustment',
            'https://www.federalreserve.gov/publications/files/large-bank-supervision-201901.pdf',
            'https://www.bis.org/publ/bcbs347.pdf',
            'https://www.imf.org/en/Publications/WP/Issues/2019/05/17/Macroprudential-Policy-and-Quantitative-Easing-46900',
            
            # Insurance Supervisors
            'https://www.eiopa.europa.eu/sites/default/files/publications/reports/eiopa-bos-20-749-final-report-on-sustainable-finance.pdf',
            'https://www.iais-web.org/page/supervisory-material/issues-papers//file/92007/issues-paper-on-climate-change-risks-to-the-insurance-sector',
            'https://content.naic.org/sites/default/files/inline-files/2021%20RBC%20Report_Final.pdf',
            
            # International Organizations  
            'https://documents1.worldbank.org/curated/en/412041519591568199/pdf/123804-WP-PUBLIC-DisasterRiskFinancing.pdf',
            'https://www.oecd.org/daf/fin/insurance/OECD-Financial-Management-of-Flood-Risk.pdf',
            'https://www.undrr.org/publication/sendai-framework-disaster-risk-reduction-2015-2030',
            
            # Think Tanks & Research Institutes
            'https://www.genevaassociation.org/sites/default/files/research-topics-document-type/pdf_public/ga2019-systemic-risk-financial-stability.pdf',
            'https://www.cesifo.org/DocDL/cesifo1_wp9234.pdf',
            'https://www.nber.org/system/files/working_papers/w28419/w28419.pdf'
        ]
    
    def search_crossref_parallel(self, query: str, max_results: int = 50) -> list:
        """Search Crossref API for academic papers"""
        papers = []
        
        try:
            url = self.api_sources['crossref']
            params = {
                'query': query,
                'rows': max_results,
                'sort': 'relevance',
                'filter': 'type:journal-article,has-full-text:true'
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            for item in data.get('message', {}).get('items', []):
                # Look for DOI and try to find PDF
                doi = item.get('DOI')
                title = ' '.join(item.get('title', []))
                
                if doi and title:
                    # Try to construct PDF URL
                    pdf_url = f"https://doi.org/{doi}"
                    
                    paper = {
                        'title': title,
                        'doi': doi,
                        'pdf_url': pdf_url,
                        'source': 'Crossref',
                        'query': query,
                        'year': item.get('published-print', {}).get('date-parts', [[0]])[0][0] or item.get('published-online', {}).get('date-parts', [[0]])[0][0],
                        'authors': [f"{author.get('given', '')} {author.get('family', '')}" for author in item.get('author', [])]
                    }
                    papers.append(paper)
                    
            logger.info(f"Crossref found {len(papers)} papers for: {query[:30]}...")
            time.sleep(1)  # Rate limiting
            
        except Exception as e:
            logger.error(f"Crossref search failed for '{query}': {str(e)}")
        
        return papers
    
    def search_openalex_parallel(self, query: str, max_results: int = 50) -> list:
        """Search OpenAlex API for papers with open access"""
        papers = []
        
        try:
            url = self.api_sources['openalex']
            params = {
                'search': query,
                'filter': 'is_oa:true,type:article',
                'per_page': max_results,
                'sort': 'cited_by_count:desc'
            }
            
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            for work in data.get('results', []):
                title = work.get('title', '')
                doi = work.get('doi', '').replace('https://doi.org/', '') if work.get('doi') else None
                
                # Look for open access PDF
                pdf_url = None
                open_access = work.get('open_access', {})
                
                if open_access.get('oa_url'):
                    pdf_url = open_access['oa_url']
                elif work.get('primary_location', {}).get('pdf_url'):
                    pdf_url = work['primary_location']['pdf_url']
                
                if pdf_url and title:
                    paper = {
                        'title': title,
                        'doi': doi,
                        'pdf_url': pdf_url,
                        'source': 'OpenAlex',
                        'query': query,
                        'year': work.get('publication_year'),
                        'cited_by_count': work.get('cited_by_count', 0)
                    }
                    papers.append(paper)
            
            logger.info(f"OpenAlex found {len(papers)} papers for: {query[:30]}...")
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"OpenAlex search failed for '{query}': {str(e)}")
        
        return papers
    
    def parallel_search_worker(self, search_batch: list) -> list:
        """Worker function for parallel searching"""
        all_papers = []
        
        for query in search_batch:
            # Multi-source search for each query
            arxiv_papers = self.search_arxiv_papers(query, max_results=20)
            semantic_papers = self.search_semantic_scholar(query, max_results=25)
            crossref_papers = self.search_crossref_parallel(query, max_results=30)
            openalex_papers = self.search_openalex_parallel(query, max_results=25)
            
            batch_papers = arxiv_papers + semantic_papers + crossref_papers + openalex_papers
            all_papers.extend(batch_papers)
            
            logger.info(f"Batch search for '{query}' found {len(batch_papers)} papers")
            
            # Rate limiting between queries
            time.sleep(2)
        
        return all_papers
    
    def industrial_scale_collection(self, target_minimum: int = 1000) -> list:
        """Industrial scale collection using parallel processing"""
        logger.info(f"🏭 Starting industrial-scale collection (target: {target_minimum}+ papers)")
        
        current_count = len(self.downloaded_papers)
        logger.info(f"Current papers: {current_count}")
        
        # Split search terms into batches for parallel processing
        batch_size = 20
        search_batches = [
            self.massive_search_terms[i:i + batch_size] 
            for i in range(0, len(self.massive_search_terms), batch_size)
        ]
        
        logger.info(f"Processing {len(self.massive_search_terms)} search terms in {len(search_batches)} parallel batches")
        
        all_papers = []
        
        # Parallel search execution
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.parallel_search_worker, batch) for batch in search_batches]
            
            for i, future in enumerate(futures):
                try:
                    batch_papers = future.result(timeout=300)  # 5 min timeout per batch
                    all_papers.extend(batch_papers)
                    logger.info(f"Completed batch {i+1}/{len(search_batches)}: {len(batch_papers)} papers")
                except Exception as e:
                    logger.error(f"Batch {i+1} failed: {str(e)}")
        
        # Add institutional PDFs
        logger.info("Adding institutional PDFs...")
        for url in self.institutional_pdfs:
            try:
                response = self.session.get(url, timeout=30)
                if response.status_code == 200 and 'pdf' in response.headers.get('content-type', '').lower():
                    paper = {
                        'title': f"Institutional Document {url.split('/')[-1]}",
                        'pdf_url': url,
                        'source': 'Institutional',
                        'content': response.content
                    }
                    all_papers.append(paper)
                time.sleep(1)
            except Exception as e:
                logger.debug(f"Institutional PDF failed: {url}")
        
        logger.info(f"Total papers found: {len(all_papers)}")
        
        # Deduplicate and rank
        unique_papers = self._deduplicate_papers(all_papers)
        ranked_papers = self._rank_papers_by_quality(unique_papers)
        
        logger.info(f"After deduplication: {len(ranked_papers)} unique papers")
        
        # Download phase with aggressive Sci-Hub usage
        logger.info("🔬 Starting download phase with Sci-Hub fallback...")
        
        needed = target_minimum - current_count
        successful_uploads = 0
        scihub_successes = 0
        failed_downloads = []
        
        for i, paper in enumerate(ranked_papers):
            if current_count + successful_uploads >= target_minimum:
                break
                
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(ranked_papers)} processed, {successful_uploads} uploaded")
            
            # Download with Sci-Hub fallback
            pdf_content = None
            used_scihub = False
            
            if 'content' in paper:
                pdf_content = paper['content']
            else:
                # Try original source
                pdf_content = self.download_pdf(paper['pdf_url'], paper['title'])
                
                # If failed, aggressive Sci-Hub attempt
                if not pdf_content:
                    pdf_content = self.download_from_scihub_for_paper(paper)
                    if pdf_content:
                        used_scihub = True
                        scihub_successes += 1
                    else:
                        failed_downloads.append(paper)
            
            if not pdf_content:
                continue
            
            # Process and upload
            try:
                pdf_metadata = self.extract_pdf_metadata(pdf_content)
                
                combined_metadata = {
                    'title': paper.get('title', ''),
                    'source': paper.get('source', ''),
                    'pdf_url': paper.get('pdf_url', ''),
                    'collection_date': datetime.now().isoformat(),
                    'scihub_used': used_scihub,
                    'industrial_collection': True,
                    'doi': paper.get('doi', ''),
                    'year': paper.get('year', ''),
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
                    if total % 100 == 0:  # Milestone logging
                        logger.info(f"🎯 MILESTONE: {total} papers collected!")
            
            except Exception as e:
                logger.error(f"Upload failed for {paper['title'][:30]}: {str(e)}")
                continue
            
            # Minimal rate limiting for high throughput
            if not used_scihub:
                time.sleep(0.5)
            else:
                time.sleep(1)  # Slightly longer for Sci-Hub
        
        final_count = current_count + successful_uploads
        
        # Final summary
        logger.info("="*80)
        logger.info("🏭 INDUSTRIAL COLLECTION COMPLETE!")
        logger.info(f"📊 Total papers collected: {final_count}")
        logger.info(f"📈 New papers added: {successful_uploads}")
        logger.info(f"🔬 Sci-Hub successes: {scihub_successes}")
        logger.info(f"❌ Failed downloads: {len(failed_downloads)}")
        logger.info(f"✅ Target achieved: {final_count >= target_minimum}")
        logger.info("="*80)
        
        return self.downloaded_papers

def main():
    """Run industrial scale collection"""
    scraper = IndustrialInsuranceScraper()
    
    # Industrial collection for 1000+ papers
    papers = scraper.industrial_scale_collection(target_minimum=1000)
    
    # Save comprehensive catalog
    scraper.save_catalog("industrial_insurance_corpus_catalog.json")
    
    print(f"🏭 Industrial collection complete!")
    print(f"📊 Total papers: {len(papers)}")
    print(f"✅ Ready for serious fine-tuning!" if len(papers) >= 1000 else f"⚠️ Target: 1000, Achieved: {len(papers)}")

if __name__ == "__main__":
    main()