#!/usr/bin/env python3
"""
Smart Collector Base Class - Content-First, Star-Based Quality Collection

Uses GitHub search API to find high-quality repositories by content patterns
and star count, then extracts relevant examples regardless of folder structure.
"""

import logging
import os
import re
import time
from typing import Dict, List, Optional, Tuple
from github import Github
from github.Repository import Repository
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart_collector")

class SmartCollectorBase:
    """Base class for content-first, star-based quality collection"""
    
    def __init__(self, domain_name: str, github_token: Optional[str] = None):
        self.domain_name = domain_name
        self.github_token = github_token or os.getenv('GITHUB_TOKEN')
        self.github_client = Github(self.github_token) if self.github_token else None
        
        if not self.github_client:
            raise ValueError("GitHub token required for smart collection")
    
    def search_repositories_by_content(self, 
                                     search_terms: List[str], 
                                     min_stars: int = 100,
                                     language: str = "python",
                                     max_repos: int = 50) -> List[Repository]:
        """Search for repositories using content patterns and star filtering"""
        repositories = []
        
        for term in search_terms:
            try:
                # GitHub search query with content and quality filters
                query = f'"{term}" stars:>{min_stars} language:{language}'
                logger.info(f"Searching: {query}")
                
                search_results = self.github_client.search_repositories(
                    query=query,
                    sort="stars",
                    order="desc"
                )
                
                # Take top repositories by stars
                for repo in search_results[:max_repos // len(search_terms)]:
                    if repo not in repositories:
                        repositories.append(repo)
                        logger.info(f"Found quality repo: {repo.full_name} ({repo.stargazers_count} stars)")
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.warning(f"Search failed for term '{term}': {str(e)}")
                continue
        
        logger.info(f"Found {len(repositories)} high-quality repositories")
        return repositories[:max_repos]
    
    def extract_content_by_patterns(self, 
                                   repo: Repository, 
                                   content_patterns: List[str],
                                   file_extensions: List[str] = ['.py'],
                                   max_files: int = 20) -> List[Dict]:
        """Extract files containing specific content patterns"""
        examples = []
        processed_files = 0
        
        try:
            # Get all files in repository (recursive)
            all_files = self._get_all_files_recursive(repo)
            
            for file_info in all_files:
                if processed_files >= max_files:
                    break
                    
                # Filter by file extension
                if not any(file_info.path.endswith(ext) for ext in file_extensions):
                    continue
                
                try:
                    # Get file content
                    content = self._get_file_content_safe(repo, file_info.path)
                    if not content:
                        continue
                    
                    # Check if content matches any patterns
                    if self._content_matches_patterns(content, content_patterns):
                        example = self._create_example_from_content(
                            content, file_info.path, repo.full_name, repo.stargazers_count
                        )
                        if example:
                            examples.append(example)
                            processed_files += 1
                            logger.debug(f"Extracted example from {file_info.path}")
                
                except Exception as e:
                    logger.debug(f"Could not process file {file_info.path}: {str(e)}")
                    continue
            
            logger.info(f"Extracted {len(examples)} examples from {repo.full_name}")
            
        except Exception as e:
            logger.error(f"Error processing repository {repo.full_name}: {str(e)}")
        
        return examples
    
    def _get_all_files_recursive(self, repo: Repository, path: str = "") -> List:
        """Recursively get all files in repository"""
        files = []
        try:
            contents = repo.get_contents(path)
            
            if not isinstance(contents, list):
                contents = [contents]
            
            for content in contents:
                if content.type == "dir":
                    # Recursively get files from subdirectory
                    files.extend(self._get_all_files_recursive(repo, content.path))
                else:
                    files.append(content)
                    
        except Exception as e:
            logger.debug(f"Could not access path {path}: {str(e)}")
        
        return files
    
    def _get_file_content_safe(self, repo: Repository, file_path: str) -> Optional[str]:
        """Safely get file content with error handling"""
        try:
            file_content = repo.get_contents(file_path)
            
            # Handle base64 encoding
            if file_content.encoding == 'base64':
                content = base64.b64decode(file_content.content).decode('utf-8')
            else:
                content = file_content.content
            
            # Skip very large files
            if len(content) > 50000:  # 50KB limit
                return None
                
            return content
            
        except Exception as e:
            logger.debug(f"Could not read file {file_path}: {str(e)}")
            return None
    
    def _content_matches_patterns(self, content: str, patterns: List[str]) -> bool:
        """Check if content contains any of the specified patterns"""
        content_lower = content.lower()
        
        for pattern in patterns:
            if pattern.lower() in content_lower:
                return True
        
        return False
    
    def _create_example_from_content(self, 
                                   content: str, 
                                   file_path: str, 
                                   repo_name: str,
                                   star_count: int) -> Optional[Dict]:
        """Create training example from content - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _create_example_from_content")
    
    def collect_domain_examples(self, 
                               target_count: int,
                               search_config: Dict) -> List[Dict]:
        """Main collection method - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement collect_domain_examples")
    
    def _assess_content_quality(self, content: str, star_count: int) -> float:
        """Assess content quality based on various factors"""
        quality_score = 0.5  # Base score
        
        # Star-based quality bonus
        if star_count > 10000:
            quality_score += 0.3
        elif star_count > 5000:
            quality_score += 0.2
        elif star_count > 1000:
            quality_score += 0.1
        
        # Content quality indicators
        quality_indicators = [
            'def ', 'class ', 'import ', 'from ',  # Python structure
            'try:', 'except:', 'finally:',         # Error handling
            '"""', "'''",                          # Documentation
            'test_', 'assert',                     # Testing
            'config', 'settings',                  # Configuration
            'logger', 'logging',                   # Logging
        ]
        
        indicator_count = sum(1 for indicator in quality_indicators if indicator in content)
        quality_score += min(0.3, indicator_count * 0.03)
        
        # Penalize very short or very long files
        line_count = len(content.split('\n'))
        if line_count < 10:
            quality_score -= 0.2
        elif line_count > 1000:
            quality_score -= 0.1
        
        return min(1.0, max(0.0, quality_score))
    
    def _generate_claude_md_prompt(self, content: str, domain: str, category: str) -> str:
        """Generate CLAUDE.md methodology prompt"""
        # Extract key information from content
        if 'class ' in content:
            pattern_type = "class-based implementation"
        elif 'def ' in content:
            pattern_type = "function-based implementation"
        elif 'import ' in content:
            pattern_type = "module integration"
        else:
            pattern_type = "configuration"
        
        return f"""### {domain.title()} Request (CLAUDE.md):
Create {pattern_type} for {category}

### EXPLORE Phase:
Analyze requirements and existing patterns.

### PLAN Phase:
Design implementation approach and structure.

### CODE Phase:"""