#!/usr/bin/env python3
"""
Universal Validation Pipeline for Qwen Multi-Domain Training

Provides quality assurance for all 6 domain collectors with comprehensive
validation including syntax checking, completeness assessment, authenticity
verification, and CLAUDE.md methodology compliance.
"""

import ast
import json
import logging
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import yaml
import bandit
from bandit.core import manager as bandit_manager
from bandit.core import config as bandit_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("universal_validation")

class SyntaxValidator:
    """Language-specific syntax validation"""
    
    def __init__(self):
        self.validators = {
            'python': self._validate_python,
            'javascript': self._validate_javascript,
            'typescript': self._validate_typescript,
            'yaml': self._validate_yaml,
            'json': self._validate_json,
            'sql': self._validate_sql,
            'bash': self._validate_bash,
            'dockerfile': self._validate_dockerfile
        }
    
    def validate(self, code: str, language: str) -> Tuple[bool, str]:
        """Validate syntax for given language"""
        if language.lower() not in self.validators:
            return True, f"Unknown language: {language}"
        
        try:
            return self.validators[language.lower()](code)
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    def _validate_python(self, code: str) -> Tuple[bool, str]:
        """Validate Python syntax using AST"""
        try:
            ast.parse(code)
            return True, "Valid Python syntax"
        except SyntaxError as e:
            return False, f"Python syntax error: {e.msg} at line {e.lineno}"
    
    def _validate_javascript(self, code: str) -> Tuple[bool, str]:
        """Validate JavaScript syntax using Node.js"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as f:
                f.write(code)
                f.flush()
                
                result = subprocess.run(
                    ['node', '--check', f.name],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    return True, "Valid JavaScript syntax"
                else:
                    return False, f"JavaScript syntax error: {result.stderr}"
        except subprocess.TimeoutExpired:
            return False, "JavaScript validation timeout"
        except Exception as e:
            return False, f"JavaScript validation error: {str(e)}"
    
    def _validate_typescript(self, code: str) -> Tuple[bool, str]:
        """Validate TypeScript syntax using tsc"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ts', delete=False) as f:
                f.write(code)
                f.flush()
                
                result = subprocess.run(
                    ['tsc', '--noEmit', '--skipLibCheck', f.name],
                    capture_output=True,
                    text=True,
                    timeout=15
                )
                
                if result.returncode == 0:
                    return True, "Valid TypeScript syntax"
                else:
                    return False, f"TypeScript syntax error: {result.stderr}"
        except subprocess.TimeoutExpired:
            return False, "TypeScript validation timeout"
        except Exception as e:
            return False, f"TypeScript validation error: {str(e)}"
    
    def _validate_yaml(self, code: str) -> Tuple[bool, str]:
        """Validate YAML syntax"""
        try:
            yaml.safe_load(code)
            return True, "Valid YAML syntax"
        except yaml.YAMLError as e:
            return False, f"YAML syntax error: {str(e)}"
    
    def _validate_json(self, code: str) -> Tuple[bool, str]:
        """Validate JSON syntax"""
        try:
            json.loads(code)
            return True, "Valid JSON syntax"
        except json.JSONDecodeError as e:
            return False, f"JSON syntax error: {str(e)}"
    
    def _validate_sql(self, code: str) -> Tuple[bool, str]:
        """Basic SQL syntax validation"""
        # Basic SQL keyword validation (can be enhanced with sqlparse)
        sql_keywords = [
            'SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE',
            'CREATE', 'ALTER', 'DROP', 'TABLE', 'INDEX', 'VIEW'
        ]
        
        code_upper = code.upper()
        has_sql_keywords = any(keyword in code_upper for keyword in sql_keywords)
        
        if not has_sql_keywords:
            return False, "No valid SQL keywords found"
        
        # Check for basic syntax patterns
        if code.count('(') != code.count(')'):
            return False, "Unmatched parentheses in SQL"
        
        return True, "Basic SQL syntax appears valid"
    
    def _validate_bash(self, code: str) -> Tuple[bool, str]:
        """Validate Bash script syntax"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
                f.write(code)
                f.flush()
                
                result = subprocess.run(
                    ['bash', '-n', f.name],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    return True, "Valid Bash syntax"
                else:
                    return False, f"Bash syntax error: {result.stderr}"
        except subprocess.TimeoutExpired:
            return False, "Bash validation timeout"
        except Exception as e:
            return False, f"Bash validation error: {str(e)}"
    
    def _validate_dockerfile(self, code: str) -> Tuple[bool, str]:
        """Validate Dockerfile syntax"""
        dockerfile_instructions = [
            'FROM', 'RUN', 'CMD', 'LABEL', 'EXPOSE', 'ENV',
            'ADD', 'COPY', 'ENTRYPOINT', 'VOLUME', 'USER', 'WORKDIR'
        ]
        
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        
        if not lines:
            return False, "Empty Dockerfile"
        
        # First instruction should be FROM
        if not lines[0].upper().startswith('FROM'):
            return False, "Dockerfile must start with FROM instruction"
        
        # Check for valid instructions
        for line in lines:
            if line.startswith('#'):  # Skip comments
                continue
            
            instruction = line.split()[0].upper()
            if instruction not in dockerfile_instructions:
                return False, f"Invalid Dockerfile instruction: {instruction}"
        
        return True, "Valid Dockerfile syntax"

class CompletenessAnalyzer:
    """Assess example completeness and production readiness"""
    
    def __init__(self):
        self.incompleteness_indicators = [
            'TODO', 'FIXME', 'XXX', 'HACK', 'BUG',
            '...', 'placeholder', 'implement this',
            'add your code here', 'fill in', 'replace with',
            'your_', 'example_', 'sample_'
        ]
        
        self.quality_indicators = [
            'error handling', 'exception', 'try:', 'catch',
            'logging', 'logger', 'documentation', 'docstring',
            'type hint', 'typing', 'validate', 'validation'
        ]
    
    def assess_completeness(self, example: Dict) -> Dict:
        """Assess example completeness with detailed scoring"""
        completion = example.get('completion', '')
        prompt = example.get('prompt', '')
        
        # Base completeness score
        completeness_score = 1.0
        issues = []
        
        # Check for incompleteness indicators
        for indicator in self.incompleteness_indicators:
            if indicator.lower() in completion.lower():
                penalty = self._calculate_penalty(indicator, completion)
                completeness_score -= penalty
                issues.append(f"Found incomplete indicator: {indicator}")
        
        # Check for quality indicators (bonus points)
        quality_bonus = 0
        for indicator in self.quality_indicators:
            if indicator.lower() in completion.lower():
                quality_bonus += 0.05
        
        completeness_score = min(1.0, max(0.0, completeness_score + quality_bonus))
        
        # Additional checks
        length_score = self._assess_length_appropriateness(prompt, completion)
        structure_score = self._assess_code_structure(completion)
        
        return {
            'completeness_score': completeness_score,
            'length_appropriateness': length_score,
            'code_structure': structure_score,
            'issues': issues,
            'quality_indicators_found': quality_bonus > 0
        }
    
    def _calculate_penalty(self, indicator: str, completion: str) -> float:
        """Calculate penalty based on indicator frequency and context"""
        count = completion.lower().count(indicator.lower())
        base_penalty = 0.2
        
        # Higher penalty for critical indicators
        critical_indicators = ['TODO', 'FIXME', 'implement this', 'placeholder']
        if indicator in critical_indicators:
            base_penalty = 0.3
        
        return min(0.5, base_penalty * count)
    
    def _assess_length_appropriateness(self, prompt: str, completion: str) -> float:
        """Assess if completion length is appropriate for prompt complexity"""
        prompt_length = len(prompt.split())
        completion_length = len(completion.split())
        
        if prompt_length == 0:
            return 0.5
        
        ratio = completion_length / prompt_length
        
        # Appropriate ratios: 2-10 (completion should be 2-10x prompt length)
        if 2 <= ratio <= 10:
            return 1.0
        elif 1 <= ratio < 2:
            return 0.7  # Too brief
        elif 10 < ratio <= 20:
            return 0.8  # Slightly verbose but acceptable
        else:
            return 0.5  # Either too brief or too verbose
    
    def _assess_code_structure(self, completion: str) -> float:
        """Assess code structure quality"""
        structure_score = 0.5  # Base score
        
        # Check for proper imports (Python)
        if any(line.strip().startswith('import ') or line.strip().startswith('from ') 
               for line in completion.split('\n')):
            structure_score += 0.1
        
        # Check for functions/classes
        if any(keyword in completion for keyword in ['def ', 'class ', 'function ']):
            structure_score += 0.1
        
        # Check for comments
        comment_patterns = ['#', '//', '/*', '"""', "'''"]
        if any(pattern in completion for pattern in comment_patterns):
            structure_score += 0.1
        
        # Check for proper indentation (basic check)
        lines = completion.split('\n')
        indented_lines = [line for line in lines if line.startswith('    ') or line.startswith('\t')]
        if indented_lines:
            structure_score += 0.1
        
        return min(1.0, structure_score)

class AuthenticityVerifier:
    """Verify if examples come from authentic production sources"""
    
    def __init__(self):
        self.authentic_domains = [
            'github.com', 'gitlab.com', 'docs.aws.amazon.com',
            'kubernetes.io', 'docs.microsoft.com', 'cloud.google.com',
            'engineering.', 'blog.', '.gov', 'stackoverflow.com',
            'apache.org', 'python.org', 'nodejs.org'
        ]
        
        self.high_quality_repos = [
            'microsoft/', 'google/', 'facebook/', 'netflix/',
            'apache/', 'kubernetes/', 'docker/', 'elastic/'
        ]
    
    def verify_authenticity(self, example: Dict) -> Dict:
        """Verify example authenticity with detailed scoring"""
        metadata = example.get('metadata', {})
        source = metadata.get('source', '')
        
        authenticity_score = 0.5  # Base score for unknown sources
        authenticity_indicators = []
        
        # Check for authentic domains
        for domain in self.authentic_domains:
            if domain in source.lower():
                authenticity_score = 0.8
                authenticity_indicators.append(f"Authentic domain: {domain}")
                break
        
        # Bonus for high-quality repositories
        for repo in self.high_quality_repos:
            if repo in source.lower():
                authenticity_score = min(1.0, authenticity_score + 0.2)
                authenticity_indicators.append(f"High-quality repository: {repo}")
                break
        
        # Check for production indicators in the code
        production_indicators = self._check_production_indicators(example)
        authenticity_score = min(1.0, authenticity_score + production_indicators['score'])
        authenticity_indicators.extend(production_indicators['indicators'])
        
        return {
            'authenticity_score': authenticity_score,
            'indicators': authenticity_indicators,
            'source_verified': authenticity_score > 0.7
        }
    
    def _check_production_indicators(self, example: Dict) -> Dict:
        """Check for indicators that suggest production-quality code"""
        completion = example.get('completion', '')
        
        production_patterns = [
            r'logging\.|logger\.',  # Logging usage
            r'try:\s*\n.*except',   # Error handling
            r'raise\s+\w+Error',    # Proper exception raising
            r'assert\s+',           # Assertions
            r'@\w+\(',             # Decorators
            r'class\s+\w+\(',      # Class definitions
            r'def\s+test_',        # Test functions
            r'import\s+os',        # System imports
            r'config\.|settings\.', # Configuration usage
        ]
        
        indicators = []
        score_bonus = 0
        
        for pattern in production_patterns:
            matches = re.findall(pattern, completion, re.MULTILINE | re.IGNORECASE)
            if matches:
                indicators.append(f"Production pattern: {pattern}")
                score_bonus += 0.05
        
        return {
            'score': min(0.3, score_bonus),  # Max 0.3 bonus
            'indicators': indicators
        }

class CLAUDEMethodologyValidator:
    """Validate CLAUDE.md methodology compliance"""
    
    def __init__(self):
        self.required_phases = ['CLAUDE.md', 'Phase']
        self.methodology_phases = [
            'EXPLORE', 'PLAN', 'CODE', 'COMMIT',
            'VALIDATION', 'REVIEW'
        ]
    
    def validate_methodology(self, example: Dict) -> Dict:
        """Validate CLAUDE.md methodology compliance"""
        prompt = example.get('prompt', '')
        completion = example.get('completion', '')
        combined_text = f"{prompt} {completion}"
        
        # Check for CLAUDE.md references
        has_claude_reference = any(
            phrase in combined_text for phrase in self.required_phases
        )
        
        # Check for methodology phases
        phases_found = []
        for phase in self.methodology_phases:
            if phase in combined_text.upper():
                phases_found.append(phase)
        
        # Calculate methodology score
        methodology_score = 0.0
        
        if has_claude_reference:
            methodology_score += 0.3
        
        if phases_found:
            # Score based on number of phases (max 0.7)
            phase_score = min(0.7, len(phases_found) * 0.15)
            methodology_score += phase_score
        
        return {
            'methodology_score': methodology_score,
            'has_claude_reference': has_claude_reference,
            'phases_found': phases_found,
            'compliant': methodology_score >= 0.5
        }

class SecurityValidator:
    """Validate code security using bandit and custom rules"""
    
    def __init__(self):
        self.security_patterns = [
            r'os\.system\(',           # Dangerous system calls
            r'eval\(',                 # Eval usage
            r'exec\(',                 # Exec usage
            r'subprocess\.call\(',     # Subprocess without shell=False
            r'shell=True',             # Shell injection risk
            r'password\s*=\s*["\'][^"\']+["\']',  # Hardcoded passwords
            r'api_key\s*=\s*["\'][^"\']+["\']',   # Hardcoded API keys
        ]
    
    def validate_security(self, example: Dict) -> Dict:
        """Validate code security"""
        completion = example.get('completion', '')
        language = example.get('metadata', {}).get('language', 'unknown')
        
        security_issues = []
        security_score = 1.0
        
        # Pattern-based security checks
        for pattern in self.security_patterns:
            matches = re.findall(pattern, completion, re.IGNORECASE)
            if matches:
                security_issues.append(f"Security risk pattern: {pattern}")
                security_score -= 0.2
        
        # Language-specific security validation
        if language.lower() == 'python':
            bandit_results = self._run_bandit_analysis(completion)
            security_issues.extend(bandit_results['issues'])
            security_score = min(security_score, bandit_results['score'])
        
        return {
            'security_score': max(0.0, security_score),
            'security_issues': security_issues,
            'is_secure': security_score >= 0.8
        }
    
    def _run_bandit_analysis(self, code: str) -> Dict:
        """Run bandit security analysis on Python code"""
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                f.flush()
                
                # Configure bandit
                conf = bandit_config.BanditConfig()
                b_mgr = bandit_manager.BanditManager(conf, 'file')
                b_mgr.discover_files([f.name])
                b_mgr.run_tests()
                
                issues = []
                security_score = 1.0
                
                for result in b_mgr.get_issue_list():
                    issues.append(f"Bandit {result.severity}: {result.text}")
                    if result.severity in ['HIGH', 'MEDIUM']:
                        security_score -= 0.3
                    else:
                        security_score -= 0.1
                
                return {
                    'score': max(0.0, security_score),
                    'issues': issues
                }
        except Exception as e:
            return {
                'score': 0.8,  # Default score if bandit fails
                'issues': [f"Bandit analysis failed: {str(e)}"]
            }

class UniversalValidationPipeline:
    """Main validation pipeline orchestrating all validators"""
    
    def __init__(self):
        self.syntax_validator = SyntaxValidator()
        self.completeness_analyzer = CompletenessAnalyzer()
        self.authenticity_verifier = AuthenticityVerifier()
        self.methodology_validator = CLAUDEMethodologyValidator()
        self.security_validator = SecurityValidator()
        
        # Validation weights for overall score calculation
        self.weights = {
            'syntax_valid': 0.25,
            'completeness_score': 0.20,
            'authenticity_score': 0.15,
            'methodology_score': 0.15,
            'security_score': 0.15,
            'educational_value': 0.10
        }
    
    def validate_example(self, example: Dict) -> Dict:
        """Comprehensive validation for any domain example"""
        validation_start = logger.info(f"Starting validation for example from {example.get('domain', 'unknown')}")
        
        # Extract basic information
        completion = example.get('completion', '')
        language = example.get('metadata', {}).get('language', 'python')
        
        # Run all validation components
        validation_results = {}
        
        # 1. Syntax validation
        syntax_valid, syntax_message = self.syntax_validator.validate(completion, language)
        validation_results['syntax_valid'] = syntax_valid
        validation_results['syntax_message'] = syntax_message
        
        # 2. Completeness analysis
        completeness_results = self.completeness_analyzer.assess_completeness(example)
        validation_results.update(completeness_results)
        
        # 3. Authenticity verification
        authenticity_results = self.authenticity_verifier.verify_authenticity(example)
        validation_results.update(authenticity_results)
        
        # 4. CLAUDE.md methodology validation
        methodology_results = self.methodology_validator.validate_methodology(example)
        validation_results.update(methodology_results)
        
        # 5. Security validation
        security_results = self.security_validator.validate_security(example)
        validation_results.update(security_results)
        
        # 6. Educational value assessment
        educational_value = self._assess_educational_value(example)
        validation_results['educational_value'] = educational_value
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_score(validation_results)
        validation_results['overall_quality'] = overall_score
        
        # Determine if example passes quality threshold
        validation_results['passes_quality_threshold'] = overall_score >= 0.75
        
        logger.info(f"Validation complete. Overall score: {overall_score:.3f}")
        
        return validation_results
    
    def _assess_educational_value(self, example: Dict) -> float:
        """Assess educational value of example"""
        prompt = example.get('prompt', '').lower()
        completion = example.get('completion', '').lower()
        
        educational_indicators = [
            'example', 'tutorial', 'guide', 'how to', 'step by step',
            'best practice', 'pattern', 'implementation', 'demonstrates',
            'shows', 'illustrates', 'explains', 'documentation'
        ]
        
        score = 0.5  # Base score
        
        for indicator in educational_indicators:
            if indicator in prompt or indicator in completion:
                score += 0.05
        
        # Bonus for comprehensive examples
        if len(completion.split()) > 100:  # Substantial content
            score += 0.1
        
        # Bonus for multi-step explanations
        if any(word in completion for word in ['first', 'second', 'then', 'next', 'finally']):
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_overall_score(self, validation_results: Dict) -> float:
        """Calculate weighted overall quality score"""
        total_score = 0.0
        
        for metric, weight in self.weights.items():
            if metric == 'syntax_valid':
                score = 1.0 if validation_results.get(metric, False) else 0.0
            else:
                score = validation_results.get(metric, 0.5)
            
            total_score += score * weight
        
        return total_score
    
    def validate_batch(self, examples: List[Dict], parallel: bool = True) -> List[Dict]:
        """Validate a batch of examples"""
        logger.info(f"Starting batch validation of {len(examples)} examples")
        
        if not parallel:
            return [self.validate_example(example) for example in examples]
        
        # For parallel processing, we'd use multiprocessing here
        # For now, implement sequential processing
        results = []
        for i, example in enumerate(examples):
            logger.info(f"Validating example {i+1}/{len(examples)}")
            try:
                result = self.validate_example(example)
                result['validation_success'] = True
            except Exception as e:
                logger.error(f"Validation failed for example {i+1}: {str(e)}")
                result = {
                    'validation_success': False,
                    'validation_error': str(e),
                    'overall_quality': 0.0,
                    'passes_quality_threshold': False
                }
            
            results.append(result)
        
        # Summary statistics
        passed = sum(1 for r in results if r.get('passes_quality_threshold', False))
        logger.info(f"Batch validation complete. {passed}/{len(examples)} examples passed quality threshold")
        
        return results

def main():
    """Example usage of the validation pipeline"""
    pipeline = UniversalValidationPipeline()
    
    # Example test case
    test_example = {
        'domain': 'code_generation',
        'category': 'api_development',
        'prompt': 'Create a REST API endpoint for user authentication',
        'completion': '''
def authenticate_user(username: str, password: str) -> dict:
    """Authenticate user and return JWT token"""
    import hashlib
    import jwt
    import datetime
    
    # Hash password for comparison
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    
    # Database lookup (mock)
    user = get_user_by_username(username)
    if not user or user.password_hash != password_hash:
        raise ValueError("Invalid credentials")
    
    # Generate JWT token
    payload = {
        'user_id': user.id,
        'username': username,
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }
    
    token = jwt.encode(payload, 'secret_key', algorithm='HS256')
    
    return {
        'token': token,
        'expires_in': 86400,
        'user': {
            'id': user.id,
            'username': username
        }
    }
        ''',
        'metadata': {
            'source': 'github.com/example/auth-service',
            'language': 'python',
            'complexity': 'medium',
            'authentic': True
        }
    }
    
    result = pipeline.validate_example(test_example)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()