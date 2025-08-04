#!/usr/bin/env python3
"""
Smart AsobaCode MCP Collector - Content-First, Community-Wide Collection

Searches for MCP, FastMCP, and terminal UI patterns across the entire GitHub ecosystem
using star-based quality filtering. Expands far beyond local repository examples.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Import base class
sys.path.append(str(Path(__file__).parent))
from smart_collector_base import SmartCollectorBase

# Import validation pipeline
sys.path.append(str(Path(__file__).parent.parent / 'validation'))
from universal_validation_pipeline import UniversalValidationPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart_asobacode_mcp")

class SmartAsobaCodeMCPCollector(SmartCollectorBase):
    """Smart collector for AsobaCode MCP patterns from community repositories"""
    
    def __init__(self, github_token: Optional[str] = None):
        super().__init__("asobacode_mcp_business_analysis", github_token)
        self.validation_pipeline = UniversalValidationPipeline()
        
        # Content-based search configuration for MCP ecosystem
        self.search_config = {
            'mcp_protocol_implementations': {
                'search_terms': [
                    'Model Context Protocol',
                    'MCP server',
                    'mcp.server',
                    'anthropic mcp',
                    'mcp-server'
                ],
                'content_patterns': [
                    'mcp', 'server', 'protocol', 'anthropic', 'context',
                    'tool', 'schema', 'manifest', 'client'
                ],
                'min_stars': 50,  # Lower threshold for newer MCP repos
                'max_repos': 40,
                'target_examples': 120
            },
            'fastmcp_patterns': {
                'search_terms': [
                    'FastMCP',
                    'from fastmcp',
                    '@app.tool',
                    'fastmcp server',
                    'app = FastMCP'
                ],
                'content_patterns': [
                    'fastmcp', 'FastMCP', '@app.tool', 'app.tool',
                    'server', 'tool', 'decorator', 'endpoint'
                ],
                'min_stars': 20,  # Even lower for FastMCP (newer framework)
                'max_repos': 30,
                'target_examples': 100
            },
            'rich_terminal_ui': {
                'search_terms': [
                    'rich.console',
                    'rich.panel',
                    'rich.progress',
                    'from rich import',
                    'textual app'
                ],
                'content_patterns': [
                    'rich', 'console', 'panel', 'progress', 'table',
                    'animation', 'spinner', 'layout', 'textual'
                ],
                'min_stars': 200,
                'max_repos': 35,
                'target_examples': 80
            },
            'cli_conversation_flows': {
                'search_terms': [
                    'click command',
                    'typer app',
                    'argparse parser',
                    'CLI interface',
                    'command line tool'
                ],
                'content_patterns': [
                    'click', 'typer', 'argparse', 'command', 'cli',
                    'parser', 'option', 'argument', 'interactive'
                ],
                'min_stars': 500,
                'max_repos': 25,
                'target_examples': 60
            },
            'multi_provider_routing': {
                'search_terms': [
                    'llm routing',
                    'model router',
                    'provider selection',
                    'anthropic openai',
                    'cost optimization llm'
                ],
                'content_patterns': [
                    'router', 'routing', 'provider', 'anthropic', 'openai',
                    'claude', 'gpt', 'cost', 'optimization', 'fallback'
                ],
                'min_stars': 100,
                'max_repos': 20,
                'target_examples': 40
            },
            'cost_tracking_systems': {
                'search_terms': [
                    'token counting',
                    'llm cost tracking',
                    'api cost monitor',
                    'usage tracking',
                    'rate limiting'
                ],
                'content_patterns': [
                    'token', 'cost', 'tracking', 'usage', 'rate',
                    'limit', 'quota', 'billing', 'monitor', 'price'
                ],
                'min_stars': 150,
                'max_repos': 15,
                'target_examples': 30
            }
        }
    
    def collect_domain_examples(self, target_count: int = 400) -> List[Dict]:
        """Collect AsobaCode MCP examples from community repositories"""
        logger.info(f"Starting smart AsobaCode MCP collection (target: {target_count})")
        
        all_examples = []
        
        for category, config in self.search_config.items():
            logger.info(f"Collecting {category} examples (target: {config['target_examples']})")
            
            # Search for repositories with MCP/FastMCP/Rich content
            repositories = self.search_repositories_by_content(
                search_terms=config['search_terms'],
                min_stars=config['min_stars'],
                max_repos=config['max_repos']
            )
            
            category_examples = []
            
            # Extract examples from each repository
            for repo in repositories:
                if len(category_examples) >= config['target_examples']:
                    break
                
                try:
                    repo_examples = self.extract_content_by_patterns(
                        repo=repo,
                        content_patterns=config['content_patterns'],
                        file_extensions=['.py', '.ts', '.js', '.yml', '.yaml', '.json'],
                        max_files=20
                    )
                    
                    # Process and validate examples
                    for example in repo_examples:
                        if len(category_examples) >= config['target_examples']:
                            break
                        
                        processed_example = self._create_mcp_example(
                            example, category
                        )
                        
                        if processed_example and self._validate_mcp_example(processed_example):
                            category_examples.append(processed_example)
                
                except Exception as e:
                    logger.warning(f"Error processing repo {repo.full_name}: {str(e)}")
                    continue
            
            logger.info(f"Collected {len(category_examples)} {category} examples")
            all_examples.extend(category_examples)
        
        logger.info(f"Total collected: {len(all_examples)} AsobaCode MCP examples")
        return all_examples[:target_count]
    
    def _create_example_from_content(self, 
                                   content: str, 
                                   file_path: str, 
                                   repo_name: str,
                                   star_count: int) -> Optional[Dict]:
        """Create example from content (required by base class)"""
        return {
            'content': content,
            'file_path': file_path,
            'repo_name': repo_name,
            'star_count': star_count,
            'quality_score': self._assess_content_quality(content, star_count)
        }
    
    def _create_mcp_example(self, raw_example: Dict, category: str) -> Optional[Dict]:
        """Create training example for AsobaCode MCP patterns"""
        content = raw_example.get('content', '')
        if len(content.strip()) < 30:
            return None
        
        # Generate category-specific prompt
        prompt = self._generate_mcp_prompt(content, category)
        
        # Assess MCP-specific complexity
        complexity = self._assess_mcp_complexity(content, category)
        
        # Extract MCP-specific features
        features = self._extract_mcp_features(content)
        
        return {
            'domain': 'asobacode_mcp_business_analysis',
            'category': category,
            'prompt': prompt,
            'completion': content,
            'metadata': {
                'source': f"github.com/{raw_example.get('repo_name', 'unknown')}",
                'file_path': raw_example.get('file_path', ''),
                'language': self._detect_language(raw_example.get('file_path', '')),
                'complexity': complexity,
                'star_count': raw_example.get('star_count', 0),
                'quality_score': raw_example.get('quality_score', 0.5),
                'features': features,
                'authority_level': 'high' if raw_example.get('star_count', 0) > 1000 else 'medium',
                'authentic': True,
                'community_validated': True  # Indicates this comes from community, not just local
            }
        }
    
    def _generate_mcp_prompt(self, content: str, category: str) -> str:
        """Generate category-specific MCP prompt"""
        prompts = {
            'mcp_protocol_implementations': 'Create MCP server implementation with protocol compliance',
            'fastmcp_patterns': 'Create FastMCP server with @app.tool decorators and validation',
            'rich_terminal_ui': 'Create Rich terminal interface with AsobaCode conversation style',
            'cli_conversation_flows': 'Create natural language CLI with interactive conversation flow',
            'multi_provider_routing': 'Create multi-provider AI routing with cost optimization',
            'cost_tracking_systems': 'Create real-time cost tracking with token usage monitoring'
        }
        
        base_prompt = prompts.get(category, 'Create AsobaCode-style MCP implementation')
        
        # Add specific details based on content patterns
        if '@app.tool' in content:
            base_prompt += ' with FastMCP tool decorators'
        elif 'rich' in content.lower() and 'console' in content.lower():
            base_prompt += ' with Rich console formatting'
        elif 'router' in content.lower() or 'routing' in content.lower():
            base_prompt += ' with intelligent routing logic'
        elif 'cost' in content.lower() or 'token' in content.lower():
            base_prompt += ' with cost optimization and tracking'
        
        return self._generate_claude_md_prompt(content, 'asobacode_mcp', base_prompt)
    
    def _assess_mcp_complexity(self, content: str, category: str) -> str:
        """Assess MCP-specific complexity"""
        line_count = len(content.split('\n'))
        
        # Category-specific complexity indicators
        complexity_indicators = {
            'mcp_protocol_implementations': ['schema', 'manifest', 'validation', 'protocol', 'client'],
            'fastmcp_patterns': ['@app.tool', 'async def', 'pydantic', 'validation', 'error'],
            'rich_terminal_ui': ['Panel', 'Progress', 'Live', 'Layout', 'Animation'],
            'cli_conversation_flows': ['click.group', 'typer.Typer', 'interactive', 'prompt', 'confirm'],
            'multi_provider_routing': ['fallback', 'retry', 'circuit', 'breaker', 'load'],
            'cost_tracking_systems': ['rate_limit', 'quota', 'billing', 'metrics', 'alert']
        }
        
        indicators = complexity_indicators.get(category, [])
        indicator_count = sum(1 for indicator in indicators if indicator.lower() in content.lower())
        
        # MCP-specific complexity scoring
        mcp_patterns = ['mcp', 'fastmcp', 'tool', 'server', 'protocol']
        mcp_count = sum(1 for pattern in mcp_patterns if pattern.lower() in content.lower())
        
        total_complexity = indicator_count + mcp_count
        
        if line_count > 200 or total_complexity > 4:
            return 'complex'
        elif line_count > 100 or total_complexity > 2:
            return 'medium'
        else:
            return 'simple'
    
    def _extract_mcp_features(self, content: str) -> Dict:
        """Extract MCP-specific features from content"""
        features = {
            'has_mcp_protocol': 'mcp' in content.lower(),
            'has_fastmcp': 'fastmcp' in content.lower() or '@app.tool' in content,
            'has_rich_ui': 'rich' in content.lower() and any(ui in content for ui in ['Console', 'Panel', 'Progress']),
            'has_cli_interface': any(cli in content.lower() for cli in ['click', 'typer', 'argparse']),
            'has_routing_logic': any(route in content.lower() for route in ['router', 'routing', 'provider', 'fallback']),
            'has_cost_tracking': any(cost in content.lower() for cost in ['cost', 'token', 'usage', 'billing']),
            'has_async_patterns': 'async def' in content or 'await ' in content,
            'has_error_handling': 'try:' in content and 'except' in content,
            'has_validation': any(val in content.lower() for val in ['validate', 'schema', 'pydantic']),
            'has_testing': any(test in content.lower() for test in ['test_', 'assert', 'pytest']),
            'function_count': len([line for line in content.split('\n') if 'def ' in line]),
            'class_count': len([line for line in content.split('\n') if 'class ' in line]),
            'tool_count': content.count('@app.tool') + content.count('@tool')
        }
        
        return features
    
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file extension"""
        if file_path.endswith('.py'):
            return 'python'
        elif file_path.endswith(('.ts', '.tsx')):
            return 'typescript'
        elif file_path.endswith(('.js', '.jsx')):
            return 'javascript'
        elif file_path.endswith(('.yml', '.yaml')):
            return 'yaml'
        elif file_path.endswith('.json'):
            return 'json'
        else:
            return 'unknown'
    
    def _validate_mcp_example(self, example: Dict) -> bool:
        """Validate MCP example with enhanced criteria"""
        try:
            # Use universal validation pipeline
            validation_result = self.validation_pipeline.validate_example(example)
            overall_quality = validation_result.get('overall_quality', 0)
            
            # MCP-specific validation
            features = example.get('metadata', {}).get('features', {})
            
            # Bonus for MCP-specific features
            mcp_bonus = 0
            if features.get('has_mcp_protocol') or features.get('has_fastmcp'):
                mcp_bonus += 0.1
            if features.get('has_rich_ui'):
                mcp_bonus += 0.05
            if features.get('has_routing_logic'):
                mcp_bonus += 0.05
            if features.get('has_cost_tracking'):
                mcp_bonus += 0.05
            
            final_score = overall_quality + mcp_bonus
            
            # Accept examples with reasonable quality for MCP domain
            return final_score >= 0.65
            
        except Exception as e:
            logger.debug(f"MCP validation failed: {str(e)}")
            return False
    
    def save_examples(self, examples: List[Dict], output_path: str):
        """Save examples to JSONL format"""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(examples)} MCP examples to {output_path}")

def main():
    """Main collection process"""
    collector = SmartAsobaCodeMCPCollector()
    
    # Collect examples
    examples = collector.collect_domain_examples(target_count=400)
    
    # Save to file
    output_path = "/home/shingai/sort/deployments/data/corpus/smart_asobacode_mcp_corpus.jsonl"
    collector.save_examples(examples, output_path)
    
    # Print detailed summary
    categories = {}
    community_examples = 0
    
    for example in examples:
        category = example.get('category', 'unknown')
        categories[category] = categories.get(category, 0) + 1
        
        if example.get('metadata', {}).get('community_validated'):
            community_examples += 1
    
    print(f"Smart AsobaCode MCP Collection Complete:")
    for category, count in categories.items():
        print(f"  {category}: {count} examples")
    print(f"Total: {len(examples)} examples")
    print(f"Community examples: {community_examples} (vs local-only before)")
    
    # Show star distribution
    star_counts = [ex.get('metadata', {}).get('star_count', 0) for ex in examples]
    if star_counts:
        avg_stars = sum(star_counts) / len(star_counts)
        max_stars = max(star_counts)
        print(f"Quality: Average {avg_stars:.0f} stars, Max {max_stars} stars")

if __name__ == "__main__":
    main()