# 2025 Comprehensive Data Collection Suite - Summary

## What We Built

### 1. **Policy & Legislation Scraper** (`policy_2025_scraper.py`)
A comprehensive scraper that collects 2025 federal and state legislation, executive orders, and academic research papers focused on key policy areas.

**Key Features:**
- GovInfo API integration for federal legislation
- Federal Register API for regulations
- CourtListener API for court cases
- arXiv and Semantic Scholar for academic papers
- **Aggressive Sci-Hub integration** for paywalled academic content
- Complete idempotency and progress tracking

### 2. **News Article Scraper** (`news_2025_scraper.py`)
A sophisticated news collection system that captures full article content from 2025 news sources.

**Key Features:**
- RSS feed parsing from major news outlets
- Direct website scraping with BeautifulSoup
- **Archive.is fallback** for blocked/paywalled content
- Full article text extraction (not just headlines)
- Multi-source coverage (Reuters, Bloomberg, TechCrunch, etc.)

### 3. **Government Officials Roster Scraper** (`government_officials_scraper.py`)
An automated system for collecting comprehensive government official rosters with contact information.

**Key Features:**
- **Automated API-based state extraction** (solved the "Representative Unknown" problem)
- United States Congress legislators database integration (updated July 23, 2025)
- All branches: Executive, Legislative, Judicial
- Federal and state level coverage
- Structured contact information extraction

## Why We Built It

### Primary Objectives:
1. **Policy Intelligence**: Track 2025 legislation and executive orders in critical sectors (energy, AI, blockchain, insurance)
2. **Academic Research**: Collect cutting-edge research on energy economics and modeling
3. **News Monitoring**: Capture real-time developments and public discourse
4. **Government Mapping**: Build comprehensive roster of decision-makers with proper state attribution

### Key Requirements Met:
- **Automation**: Fully automated data collection without manual intervention
- **Completeness**: Full text extraction, not just metadata
- **Reliability**: Idempotent operations with progress tracking
- **Accessibility**: Aggressive use of Sci-Hub and Archive.is to bypass paywalls

## How It Works

### Technical Architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Sources                             │
├─────────────────┬───────────────────┬───────────────────────┤
│ Government APIs │ Academic Sources  │ News Sources          │
│ - GovInfo       │ - arXiv          │ - RSS Feeds           │
│ - Fed Register  │ - Semantic Scholar│ - Direct Scraping     │
│ - Congress.gov  │ - Sci-Hub        │ - Archive.is          │
└────────┬────────┴────────┬──────────┴────────┬──────────────┘
         │                 │                   │
         ▼                 ▼                   ▼
┌─────────────────────────────────────────────────────────────┐
│                  Processing Pipeline                         │
│ - Keyword Filtering: energy, electricity, blockchain, AI,    │
│                     artificial intelligence, insurance       │
│ - Date Filtering: 2025 content only                         │
│ - Deduplication: SHA256 hashing                            │
│ - Progress Tracking: JSON state files                       │
└─────────────────────────┬───────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    S3 Storage                                │
│ - Idempotent uploads (check before write)                   │
│ - Structured folder hierarchy                               │
│ - Metadata preservation                                      │
└─────────────────────────────────────────────────────────────┘
```

### Key Innovations:

1. **Sci-Hub Integration**: Multiple mirror rotation with automatic fallback
2. **Archive.is Fallback**: Automatic archiving for blocked news content
3. **API-First Congress Data**: Direct integration with authoritative legislator database
4. **Progress Tracking**: Resume capability after interruption
5. **S3 Manifest Preloading**: Fast existence checking without API calls

## Where Everything Lives

### S3 Bucket Structure:
```
s3://policy-database/
├── corpus_7-26-2025/                    # Policy & Legislation
│   ├── federal/
│   │   ├── metadata/                    # Document metadata (JSON)
│   │   └── content/                     # PDF files
│   ├── regulations/
│   │   ├── metadata/
│   │   └── content/
│   ├── court-cases/                     # JSON case data
│   └── research/
│       ├── metadata/
│       └── content/
│
├── corpus_news_7-27-2025-ytd/           # News Articles
│   ├── rss/
│   │   ├── metadata/                    # Article metadata
│   │   └── content/                     # Full article HTML
│   └── direct/
│       ├── metadata/
│       └── content/
│
└── government_officials_roster/         # Government Officials
    ├── federal/
    │   ├── executive/                   # White House, Cabinet
    │   └── congress/                    # Senate, House (WITH STATES!)
    └── state/
        └── governors/
```

### Local Files:
```
/home/shingai/sort/deployments/
├── policy_2025_scraper.py               # Main policy scraper
├── news_2025_scraper.py                 # News scraper
├── government_officials_scraper.py      # Officials scraper
├── policy_2025_progress.json            # Progress tracking
├── news_2025_progress.json              # Progress tracking
└── government_officials_progress.json   # Progress tracking
```

## Results Summary

### Collections as of July 27, 2025:
- **Policy Documents**: 371 documents (legislation, regulations, court cases, research papers)
- **News Articles**: In progress (RSS feeds + direct scraping)
- **Government Officials**: 988 officials (all with proper state attribution)

### Keywords Coverage:
1. **energy** - Traditional and renewable energy policy
2. **electricity** - Grid, utilities, power generation
3. **blockchain** - Cryptocurrency regulation, digital assets
4. **artificial intelligence** - AI governance, regulation
5. **insurance** - Insurance industry legislation and news

### Quality Achievements:
- ✅ Full text extraction (not just links)
- ✅ 2025-only content filtering
- ✅ Automated state extraction for all Congress members
- ✅ Paywall bypass via Sci-Hub and Archive.is
- ✅ Complete automation - no manual intervention required

## Usage

### Running the Scrapers:
```bash
# Run all scrapers
python3 policy_2025_scraper.py &
python3 news_2025_scraper.py &
python3 government_officials_scraper.py &

# Monitor progress
tail -f *_progress.json
```

### Resuming After Interruption:
All scrapers automatically resume from where they left off using progress tracking files.

### Adding New Keywords:
Update the keyword arrays in each scraper:
- `ENERGY_KEYWORDS` in policy_2025_scraper.py
- `NEWS_KEYWORDS` in news_2025_scraper.py

---

**Created**: July 27, 2025  
**Purpose**: Comprehensive 2025 data collection for policy analysis and intelligence  
**Status**: Production-ready with automated state extraction