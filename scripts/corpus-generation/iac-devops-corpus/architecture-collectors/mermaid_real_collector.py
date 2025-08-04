#!/usr/bin/env python3
"""
Real-World Mermaid Architecture Diagram Collector
Following CLAUDE.md principle: EXCLUSIVE REAL-WORLD DATA FOR MODEL TRAINING

Collects exclusively from authentic sources:
- Production architecture documentation
- System design diagrams from major repositories
- Real microservices and cloud architecture diagrams
- Enterprise documentation patterns
- Technical specification diagrams
"""

import os
import json
import time
import requests
import subprocess
import tempfile
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Optional, Set, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MermaidRealCollector:
    """Collects Mermaid diagram examples exclusively from real-world sources"""
    
    def __init__(self, output_dir: str = "mermaid_real_corpus"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.collected_hashes: Set[str] = set()
        self.examples = []
        
        # Rate limiting
        self.request_count = 0
        self.last_request_time = 0
        
        # Target repositories with quality Mermaid architecture diagrams
        self.target_repos = [
            # Official projects with excellent documentation
            'mermaid-js/mermaid',
            'kubernetes/kubernetes',
            'docker/docker',
            'microsoft/vscode',
            'facebook/react',
            'golang/go',
            'rust-lang/rust',
            
            # Architecture documentation examples
            'mehmetozkaya/Design-Microservices-Architecture-with-Patterns-Principles',
            'miliariadnane/demo-microservices',
            'imjoseangel/microservice-arquitecture',
            'philippemerle/KubeDiagrams',
            'sauravhaloi/mermaid',
            
            # System design repositories
            'donnemartin/system-design-primer',
            'ashishps1/awesome-system-design-resources',
            'karanpratapsingh/system-design',
            
            # Cloud native projects
            'cncf/landscape',
            'argoproj/argo-cd',
            'istio/istio',
            'envoyproxy/envoy',
            'prometheus/prometheus',
            
            # Enterprise projects with documentation
            'Netflix/conductor',
            'airbnb/lottie',
            'uber/cadence',
            'elastic/elasticsearch',
            
            # DevOps and infrastructure
            'hashicorp/terraform',
            'ansible/ansible',
            'jenkins-x/jx',
            'spinnaker/spinnaker',
            
            # Specific documentation projects
            'tldr-pages/tldr',
            'awesome-selfhosted/awesome-selfhosted',
            'public-apis/public-apis',
        ]
        
        # Local repositories to explore
        self.local_sources = [
            '/home/shingai/api',
            '/home/shingai/sort/deployments',
            '/home/shingai/sort/ona-front-end'
        ]

    def collect_real_mermaid_examples(self, target_count: int = 200) -> List[Dict]:
        """Collect real-world Mermaid diagram examples"""
        logger.info(f"Starting real-world Mermaid collection targeting {target_count} examples")
        
        all_examples = []
        
        # Strategy 1: Local repository mining for Mermaid diagrams
        local_examples = self._collect_from_local_repos()
        all_examples.extend(local_examples)
        logger.info(f"Collected {len(local_examples)} from local repositories")
        
        # Strategy 2: Clone documentation-rich repositories
        repo_examples = self._collect_from_mermaid_repos()
        all_examples.extend(repo_examples)
        logger.info(f"Collected {len(repo_examples)} from Mermaid repositories")
        
        # Strategy 3: Official documentation sources
        official_examples = self._collect_from_official_sources()
        all_examples.extend(official_examples)
        logger.info(f"Collected {len(official_examples)} from official sources")
        
        # Deduplicate and clean
        deduplicated = self._deduplicate_examples(all_examples)
        logger.info(f"Deduplicated to {len(deduplicated)} unique examples")
        
        # Filter for quality and return top examples
        quality_examples = [ex for ex in deduplicated if self._is_quality_example(ex)]
        
        # Sort by complexity and authenticity, take best examples
        sorted_examples = sorted(quality_examples, 
                               key=lambda x: (len(x['completion']), x['metadata'].get('stars', 0)), 
                               reverse=True)
        
        final_examples = sorted_examples[:target_count]
        logger.info(f"Final corpus: {len(final_examples)} high-quality Mermaid examples")
        
        return final_examples

    def _collect_from_local_repos(self) -> List[Dict]:
        """Extract Mermaid diagram examples from local repositories"""
        examples = []
        
        for source_path in self.local_sources:
            if not os.path.exists(source_path):
                continue
                
            logger.info(f"Mining Mermaid examples from {source_path}")
            path_examples = self._extract_from_path(source_path)
            examples.extend(path_examples)
            
        return examples

    def _extract_from_path(self, path: str) -> List[Dict]:
        """Extract Mermaid diagrams from a given path"""
        examples = []
        path_obj = Path(path)
        
        # Find Markdown files that might contain Mermaid diagrams
        markdown_patterns = [
            "**/*.md",
            "**/*.markdown",
            "**/README*",
            "**/ARCHITECTURE*",
            "**/DESIGN*",
            "**/docs/**/*.md",
            "**/documentation/**/*.md",
        ]
        
        for pattern in markdown_patterns:
            for file_path in path_obj.glob(pattern):
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        # Extract Mermaid diagrams from markdown
                        diagram_examples = self._extract_mermaid_from_markdown(content, str(file_path))
                        examples.extend(diagram_examples)
                                
                    except Exception as e:
                        logger.debug(f"Error reading {file_path}: {e}")
                        continue
                        
        return examples

    def _extract_mermaid_from_markdown(self, content: str, file_path: str) -> List[Dict]:
        """Extract Mermaid diagrams from Markdown content"""
        examples = []
        
        # Find all Mermaid code blocks
        mermaid_pattern = r'```mermaid\n(.*?)\n```'
        matches = re.findall(mermaid_pattern, content, re.DOTALL)
        
        # Also check for mermaid in HTML comments (some docs use this)
        html_mermaid_pattern = r'<!--\s*mermaid\s*\n(.*?)\n-->'
        html_matches = re.findall(html_mermaid_pattern, content, re.DOTALL)
        
        all_matches = matches + html_matches
        
        for diagram_content in all_matches:
            if self._is_quality_mermaid_content(f"```mermaid\n{diagram_content}\n```"):
                example = self._create_mermaid_example(
                    f"```mermaid\n{diagram_content}\n```", 
                    file_path
                )
                if example:
                    examples.append(example)
                    
        return examples

    def _collect_from_mermaid_repos(self) -> List[Dict]:
        """Clone and extract from documentation-rich repositories"""
        examples = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for repo in self.target_repos[:10]:  # Limit to avoid timeout
                try:
                    logger.info(f"Cloning {repo}")
                    clone_path = Path(temp_dir) / repo.replace('/', '_')
                    
                    # Clone with depth limit and file size filter
                    result = subprocess.run([
                        'git', 'clone', '--depth', '1', '--filter=blob:limit=500k',
                        f'https://github.com/{repo}.git', str(clone_path)
                    ], capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        repo_examples = self._extract_from_path(str(clone_path))
                        # Add repository metadata
                        for example in repo_examples:
                            example['metadata']['source_repo'] = repo
                        examples.extend(repo_examples)
                        
                    time.sleep(4)  # Rate limiting
                    
                except Exception as e:
                    logger.debug(f"Error cloning {repo}: {e}")
                    continue
                    
        return examples

    def _collect_from_official_sources(self) -> List[Dict]:
        """Collect from official documentation repositories"""
        examples = []
        
        # Official documentation repositories
        official_sources = [
            ('mermaid-js/mermaid', 'develop'),
            ('kubernetes/kubernetes', 'master'),
            ('docker/docker', 'master'),
            ('microsoft/vscode', 'main')
        ]
        
        for repo, branch in official_sources:
            try:
                logger.info(f"Fetching official documentation from {repo}")
                # Get repository file listing
                api_url = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
                response = requests.get(api_url, timeout=20)
                
                if response.status_code == 200:
                    tree = response.json()
                    
                    # Find documentation files
                    doc_files = [
                        item for item in tree.get('tree', [])
                        if item['type'] == 'blob' and (
                            item['path'].endswith('.md') or
                            'README' in item['path'] or
                            'ARCHITECTURE' in item['path'] or
                            'DESIGN' in item['path'] or
                            '/docs/' in item['path'] or
                            '/documentation/' in item['path']
                        )
                    ]
                    
                    for file_info in doc_files[:30]:  # Limit files per repo
                        try:
                            # Fetch file content
                            file_url = f"https://raw.githubusercontent.com/{repo}/{branch}/{file_info['path']}"
                            file_response = requests.get(file_url, timeout=15)
                            
                            if file_response.status_code == 200:
                                content = file_response.text
                                diagram_examples = self._extract_mermaid_from_markdown(
                                    content,
                                    f"{repo}/{file_info['path']}",
                                )
                                for example in diagram_examples:
                                    example['metadata']['source_repo'] = repo
                                examples.extend(diagram_examples)
                                        
                            time.sleep(1)  # Rate limiting
                            
                        except Exception as e:
                            logger.debug(f"Error fetching {file_info['path']}: {e}")
                            continue
                            
                time.sleep(4)  # Rate limiting between repos
                
            except Exception as e:
                logger.debug(f"Error accessing {repo}: {e}")
                continue
                
        return examples

    def _is_quality_mermaid_content(self, content: str) -> bool:
        """Check if Mermaid content meets quality standards"""
        # Remove the mermaid wrapper for analysis
        inner_content = content
        if '```mermaid' in content:
            match = re.search(r'```mermaid\n(.*?)\n```', content, re.DOTALL)
            if match:
                inner_content = match.group(1)
        
        # Size checks
        if len(inner_content) < 50 or len(inner_content) > 30000:
            return False
            
        content_lower = inner_content.lower()
        
        # Must contain Mermaid diagram indicators
        mermaid_indicators = [
            'graph', 'sequencediagram', 'classDiagram', 'stateDiagram',
            'erDiagram', 'gantt', 'pie', 'flowchart', 'mindmap',
            'timeline', 'gitgraph', 'journey', 'architecture',
            '-->', '---', '==>', '-.->',
            'participant', 'actor', 'subgraph'
        ]
        
        if not any(indicator.lower() in content_lower for indicator in mermaid_indicators):
            return False
            
        # Avoid template/example placeholders
        bad_patterns = [
            'your service', 'your component', 'your database',
            'add your', 'replace this', 'template here',
            '%% add', 'todo:', 'fixme:'
        ]
        if any(pattern in content_lower for pattern in bad_patterns):
            return False
            
        # Must have some structure (multiple lines with content)
        meaningful_lines = [line.strip() for line in inner_content.split('\n') 
                          if line.strip() and not line.strip().startswith('%%')]
        if len(meaningful_lines) < 4:
            return False
            
        # Should have nodes/components (not just empty diagram)
        node_patterns = [r'\[.*?\]', r'\(.*?\)', r'\{.*?\}', r'".*?"']
        has_nodes = any(re.search(pattern, inner_content) for pattern in node_patterns)
        if not has_nodes:
            return False
            
        return True

    def _create_mermaid_example(self, content: str, file_path: str, repo_name: str = None) -> Optional[Dict]:
        """Create a training example from Mermaid content"""
        try:
            # Generate content hash for deduplication
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in self.collected_hashes:
                return None
            self.collected_hashes.add(content_hash)
            
            # Clean and prepare content
            cleaned_content = self._clean_mermaid_content(content)
            
            # Extract inner diagram content
            inner_content = cleaned_content
            if '```mermaid' in cleaned_content:
                match = re.search(r'```mermaid\n(.*?)\n```', cleaned_content, re.DOTALL)
                if match:
                    inner_content = match.group(1)
            
            # Determine diagram type and category
            category = self._categorize_mermaid_content(inner_content)
            language = "mermaid"
            
            # Generate appropriate prompt
            prompt = self._generate_mermaid_prompt(inner_content, category, language)
            
            # Format completion (already includes ```mermaid wrapper)
            completion = cleaned_content
            
            return {
                "prompt": prompt,
                "completion": completion,
                "metadata": {
                    "source": "real_production_repository" if repo_name else "real_local_repository",
                    "file_path": file_path,
                    "category": category,
                    "authentic": True,
                    "language": language,
                    "source_repo": repo_name or "local",
                    "diagram_type": self._determine_diagram_type(inner_content),
                    "architecture_patterns": self._extract_architecture_patterns(inner_content),
                    "complexity": self._determine_diagram_complexity(inner_content)
                }
            }
            
        except Exception as e:
            logger.debug(f"Error creating example from {file_path}: {e}")
            return None

    def _clean_mermaid_content(self, content: str) -> str:
        """Clean Mermaid content while preserving authenticity"""
        lines = content.split('\n')
        clean_lines = []
        
        for line in lines:
            # Remove excessively long lines
            if len(line) > 2000:
                continue
                
            # Basic sanitization of sensitive data
            line = re.sub(r'\b\d{12}\b', '<ACCOUNT_ID>', line)
            line = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '<EMAIL>', line)
            line = re.sub(r'(password|secret|token|key)[\s]*[:=][\s]*["\'][^"\']+["\']', 
                         r'\1: "<REDACTED>"', line, flags=re.IGNORECASE)
            
            clean_lines.append(line.rstrip())
            
        return '\n'.join(clean_lines).strip()

    def _categorize_mermaid_content(self, content: str) -> str:
        """Categorize Mermaid content by diagram type"""
        content_lower = content.lower()
        
        if any(kw in content_lower for kw in ['graph', 'flowchart']) and 'subgraph' in content_lower:
            return 'architecture_flowchart'
        elif 'sequencediagram' in content_lower:
            return 'sequence_diagram'
        elif 'classdiagram' in content_lower:
            return 'class_diagram'
        elif 'statediagram' in content_lower:
            return 'state_diagram'
        elif 'erdiagram' in content_lower:
            return 'entity_relationship'
        elif 'gantt' in content_lower:
            return 'gantt_chart'
        elif 'pie' in content_lower:
            return 'pie_chart'
        elif 'journey' in content_lower:
            return 'user_journey'
        elif 'gitgraph' in content_lower:
            return 'git_flow'
        elif 'mindmap' in content_lower:
            return 'mindmap'
        elif 'timeline' in content_lower:
            return 'timeline'
        elif 'architecture' in content_lower:
            return 'architecture_diagram'
        elif any(kw in content_lower for kw in ['graph', 'flowchart']):
            return 'flowchart'
        else:
            return 'general_diagram'

    def _determine_diagram_type(self, content: str) -> str:
        """Determine the specific type of diagram"""
        content_lower = content.lower()
        
        if 'td' in content_lower or 'tb' in content_lower:
            return 'top_down'
        elif 'lr' in content_lower:
            return 'left_right'
        elif 'rl' in content_lower:
            return 'right_left'
        elif 'bt' in content_lower:
            return 'bottom_top'
        else:
            return 'default'

    def _extract_architecture_patterns(self, content: str) -> List[str]:
        """Extract architecture patterns from diagram content"""
        patterns = []
        content_lower = content.lower()
        
        # Microservices patterns
        if any(ms in content_lower for ms in ['service', 'microservice', 'api gateway']):
            patterns.append('microservices')
            
        # Kubernetes patterns
        if any(k8s in content_lower for k8s in ['pod', 'service', 'ingress', 'deployment', 'kubernetes', 'k8s']):
            patterns.append('kubernetes')
            
        # Database patterns
        if any(db in content_lower for db in ['database', 'db', 'postgres', 'mysql', 'mongodb', 'redis']):
            patterns.append('database')
            
        # CI/CD patterns
        if any(cicd in content_lower for cicd in ['pipeline', 'ci/cd', 'deploy', 'build', 'test']):
            patterns.append('cicd_pipeline')
            
        # Cloud patterns
        if any(cloud in content_lower for cloud in ['aws', 'azure', 'gcp', 'cloud', 's3', 'ec2']):
            patterns.append('cloud_architecture')
            
        # Load balancing
        if any(lb in content_lower for lb in ['load balancer', 'lb', 'nginx', 'haproxy']):
            patterns.append('load_balancing')
            
        # Message queue patterns
        if any(mq in content_lower for mq in ['queue', 'kafka', 'rabbitmq', 'sqs', 'pubsub']):
            patterns.append('message_queue')
            
        return patterns

    def _determine_diagram_complexity(self, content: str) -> str:
        """Determine the complexity level of the diagram"""
        # Count various elements
        node_count = len(re.findall(r'\[.*?\]|\(.*?\)|\{.*?\}', content))
        edge_count = content.count('-->') + content.count('---') + content.count('==>') + content.count('-.->') 
        subgraph_count = content.lower().count('subgraph')
        line_count = len([line for line in content.split('\n') if line.strip()])
        
        # Calculate complexity score
        complexity_score = (node_count * 1) + (edge_count * 0.5) + (subgraph_count * 3) + (line_count * 0.2)
        
        if complexity_score >= 50:
            return 'enterprise'
        elif complexity_score >= 20:
            return 'production'
        else:
            return 'basic'

    def _generate_mermaid_prompt(self, content: str, category: str, language: str) -> str:
        """Generate appropriate prompt for Mermaid content"""
        content_lower = content.lower()
        
        if category == 'architecture_flowchart':
            if 'microservice' in content_lower:
                return "Create a Mermaid diagram showing microservices architecture with service dependencies"
            elif 'kubernetes' in content_lower or 'k8s' in content_lower:
                return "Create a Mermaid diagram illustrating Kubernetes deployment architecture"
            elif 'database' in content_lower:
                return "Create a Mermaid diagram showing system architecture with database connections"
            else:
                return "Create a Mermaid diagram depicting system architecture and component relationships"
                
        elif category == 'sequence_diagram':
            if 'api' in content_lower:
                return "Create a Mermaid sequence diagram showing API request flow"
            elif 'auth' in content_lower:
                return "Create a Mermaid sequence diagram illustrating authentication flow"
            else:
                return "Create a Mermaid sequence diagram showing system interactions"
                
        elif category == 'class_diagram':
            return "Create a Mermaid class diagram showing object-oriented design structure"
            
        elif category == 'state_diagram':
            return "Create a Mermaid state diagram illustrating state transitions"
            
        elif category == 'entity_relationship':
            return "Create a Mermaid ER diagram showing database schema relationships"
            
        elif category == 'gantt_chart':
            return "Create a Mermaid Gantt chart for project timeline visualization"
            
        elif category == 'user_journey':
            return "Create a Mermaid user journey diagram mapping user interactions"
            
        elif category == 'flowchart':
            if 'deploy' in content_lower or 'pipeline' in content_lower:
                return "Create a Mermaid flowchart showing deployment pipeline process"
            else:
                return "Create a Mermaid flowchart illustrating process flow"
                
        else:
            return "Create a Mermaid diagram for technical documentation"

    def _deduplicate_examples(self, examples: List[Dict]) -> List[Dict]:
        """Remove duplicate examples based on content similarity"""
        seen_hashes = set()
        unique_examples = []
        
        for example in examples:
            # Create hash from completion content
            content_hash = hashlib.md5(example['completion'].encode()).hexdigest()
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_examples.append(example)
                
        return unique_examples

    def _is_quality_example(self, example: Dict) -> bool:
        """Final quality check for examples"""
        completion = example['completion']
        
        # Must have reasonable length
        if len(completion) < 100 or len(completion) > 35000:
            return False
            
        # Must contain actual diagram content
        if completion.count('\n') < 4:
            return False
            
        # Must be Mermaid-related
        if '```mermaid' not in completion and 'graph' not in completion.lower():
            return False
            
        return True

    def save_corpus(self, examples: List[Dict], filename: str = "mermaid_architecture_corpus.jsonl") -> None:
        """Save examples to JSONL corpus file"""
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                json.dump(example, f, ensure_ascii=False)
                f.write('\n')
                
        logger.info(f"Saved {len(examples)} Mermaid examples to {output_path}")

def main():
    """Collect Mermaid diagram examples for Mistral 7B training"""
    collector = MermaidRealCollector()
    
    # Collect examples
    examples = collector.collect_real_mermaid_examples(target_count=200)
    
    # Save to corpus
    collector.save_corpus(examples)
    
    # Print summary
    categories = {}
    patterns = {}
    complexity = {}
    for example in examples:
        cat = example['metadata']['category']
        comp = example['metadata']['complexity']
        categories[cat] = categories.get(cat, 0) + 1
        complexity[comp] = complexity.get(comp, 0) + 1
        
        for pattern in example['metadata'].get('architecture_patterns', []):
            patterns[pattern] = patterns.get(pattern, 0) + 1
    
    print(f"\nMermaid Corpus Summary:")
    print(f"Total examples: {len(examples)}")
    print("Categories:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
    print("Architecture Patterns:")
    for pattern, count in sorted(patterns.items()):
        print(f"  {pattern}: {count}")
    print("Complexity:")
    for comp, count in sorted(complexity.items()):
        print(f"  {comp}: {count}")

if __name__ == "__main__":
    main()