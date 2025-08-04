#!/usr/bin/env python3
"""
TDD Tests for Mermaid Architecture Diagram Collector
Following CLAUDE.md principle: Write failing tests first, then implement
"""

import pytest
import tempfile
import json
from pathlib import Path
from mermaid_real_collector import MermaidRealCollector

class TestMermaidRealCollector:
    """Test cases for Mermaid diagram collection following CLAUDE.md TDD approach"""
    
    def setup_method(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.collector = MermaidRealCollector(output_dir=self.temp_dir)
    
    def test_mermaid_collector_initialization(self):
        """Test collector initializes with required repositories"""
        # Should target production repositories with architecture documentation
        assert 'kubernetes/kubernetes' in self.collector.target_repos
        assert 'docker/docker' in self.collector.target_repos
        assert 'microsoft/vscode' in self.collector.target_repos
        
        # Should have empty collected examples initially
        assert len(self.collector.examples) == 0
        assert len(self.collector.collected_hashes) == 0

    def test_identifies_quality_mermaid_content(self):
        """Test quality detection for Mermaid diagram content"""
        # Valid flowchart diagram
        valid_flowchart = """
```mermaid
graph TD
    A[Load Balancer] --> B[Web Server 1]
    A --> C[Web Server 2]
    B --> D[Database Primary]
    C --> D
    D --> E[Database Replica]
    B --> F[Cache Layer]
    C --> F
```
"""
        assert self.collector._is_quality_mermaid_content(valid_flowchart) == True
        
        # Valid sequence diagram
        valid_sequence = """
```mermaid
sequenceDiagram
    participant Client
    participant API Gateway
    participant Auth Service
    participant Backend Service
    participant Database
    
    Client->>API Gateway: POST /api/login
    API Gateway->>Auth Service: Validate credentials
    Auth Service->>Database: Query user
    Database-->>Auth Service: User data
    Auth Service-->>API Gateway: JWT token
    API Gateway-->>Client: 200 OK + token
```
"""
        assert self.collector._is_quality_mermaid_content(valid_sequence) == True
        
        # Valid architecture diagram
        valid_architecture = """
```mermaid
graph TB
    subgraph "Frontend"
        UI[React UI]
        Mobile[Mobile App]
    end
    
    subgraph "API Layer"
        Gateway[API Gateway]
        Auth[Auth Service]
    end
    
    subgraph "Backend Services"
        Users[User Service]
        Orders[Order Service]
        Inventory[Inventory Service]
    end
    
    subgraph "Data Layer"
        DB[(PostgreSQL)]
        Cache[(Redis)]
        Queue[Message Queue]
    end
    
    UI --> Gateway
    Mobile --> Gateway
    Gateway --> Auth
    Gateway --> Users
    Gateway --> Orders
    Orders --> Inventory
    Users --> DB
    Orders --> DB
    Inventory --> DB
    Users --> Cache
    Orders --> Queue
```
"""
        assert self.collector._is_quality_mermaid_content(valid_architecture) == True
        
        # Invalid content (too short)
        invalid_short = "```mermaid\ngraph TD\nA-->B\n```"
        assert self.collector._is_quality_mermaid_content(invalid_short) == False
        
        # Invalid content (template placeholder)
        invalid_template = """
```mermaid
graph TD
    A[Your Service] --> B[Your Database]
    %% Add your components here
```
"""
        assert self.collector._is_quality_mermaid_content(invalid_template) == False

    def test_categorizes_mermaid_content_correctly(self):
        """Test proper categorization of different Mermaid diagram types"""
        flowchart = "graph TD\n    A[Service] --> B[Database]"
        sequence = "sequenceDiagram\n    Client->>Server: Request"
        class_diagram = "classDiagram\n    class User {\n        +name: string\n    }"
        state = "stateDiagram-v2\n    [*] --> Active"
        gantt = "gantt\n    title Project Timeline\n    section Phase 1"
        
        assert self.collector._categorize_mermaid_content(flowchart) == "architecture_flowchart"
        assert self.collector._categorize_mermaid_content(sequence) == "sequence_diagram"
        assert self.collector._categorize_mermaid_content(class_diagram) == "class_diagram"
        assert self.collector._categorize_mermaid_content(state) == "state_diagram"
        assert self.collector._categorize_mermaid_content(gantt) == "gantt_chart"

    def test_detects_architecture_patterns(self):
        """Test detection of specific architecture patterns in diagrams"""
        microservices = """
graph TD
    Gateway[API Gateway] --> UserService[User Service]
    Gateway --> OrderService[Order Service]
    UserService --> UserDB[(User DB)]
    OrderService --> OrderDB[(Order DB)]
"""
        
        kubernetes = """
graph TD
    Ingress[Ingress Controller] --> Service[K8s Service]
    Service --> Pod1[Pod 1]
    Service --> Pod2[Pod 2]
    Pod1 --> ConfigMap[ConfigMap]
"""
        
        cicd = """
graph LR
    Git[Git Push] --> CI[CI Pipeline]
    CI --> Build[Build]
    Build --> Test[Test]
    Test --> Deploy[Deploy]
"""
        
        patterns = self.collector._extract_architecture_patterns(microservices)
        assert "microservices" in patterns
        
        patterns = self.collector._extract_architecture_patterns(kubernetes)
        assert "kubernetes" in patterns
        
        patterns = self.collector._extract_architecture_patterns(cicd)
        assert "cicd_pipeline" in patterns

    def test_generates_appropriate_prompts(self):
        """Test prompt generation for different Mermaid content types"""
        arch_content = "graph TD\n    LB[Load Balancer] --> Web[Web Tier]\n    Web --> App[App Tier]\n    App --> DB[Database]"
        sequence_content = "sequenceDiagram\n    Client->>API: Request\n    API->>DB: Query"
        
        arch_prompt = self.collector._generate_mermaid_prompt(arch_content, "architecture_flowchart", "mermaid")
        sequence_prompt = self.collector._generate_mermaid_prompt(sequence_content, "sequence_diagram", "mermaid")
        
        assert "architecture" in arch_prompt.lower()
        assert "sequence" in sequence_prompt.lower()
        assert len(arch_prompt) > 20
        assert len(sequence_prompt) > 20

    def test_creates_valid_jsonl_examples(self):
        """Test creation of valid JSONL training examples"""
        content = """
```mermaid
graph TD
    subgraph "Production Environment"
        LB[Load Balancer]
        Web1[Web Server 1]
        Web2[Web Server 2]
        DB[(PostgreSQL)]
        Cache[(Redis)]
    end
    
    LB --> Web1
    LB --> Web2
    Web1 --> DB
    Web2 --> DB
    Web1 --> Cache
    Web2 --> Cache
```
"""
        example = self.collector._create_mermaid_example(content, "/docs/architecture.md", "kubernetes/kubernetes")
        
        # Should create valid example structure
        assert example is not None
        assert "prompt" in example
        assert "completion" in example
        assert "metadata" in example
        
        # Metadata should include required fields
        metadata = example["metadata"]
        assert metadata["source"] == "real_production_repository"
        assert metadata["category"] == "architecture_flowchart"
        assert metadata["authentic"] == True
        assert metadata["language"] == "mermaid"
        assert metadata["source_repo"] == "kubernetes/kubernetes"

    def test_extracts_from_markdown_files(self):
        """Test extraction of Mermaid diagrams from Markdown documentation"""
        markdown_content = """
# System Architecture

Our system uses a microservices architecture as shown below:

```mermaid
graph TB
    subgraph "Frontend"
        UI[Web UI]
        Mobile[Mobile App]
    end
    
    subgraph "Backend"
        API[API Gateway]
        Auth[Auth Service]
        User[User Service]
    end
    
    UI --> API
    Mobile --> API
    API --> Auth
    API --> User
```

## Database Design

The following diagram shows our database relationships:

```mermaid
erDiagram
    USER ||--o{ ORDER : places
    ORDER ||--|{ ORDER_ITEM : contains
    PRODUCT ||--o{ ORDER_ITEM : includes
```
"""
        
        examples = self.collector._extract_mermaid_from_markdown(markdown_content, "/docs/architecture.md")
        assert len(examples) == 2
        assert any("microservices" in ex["prompt"].lower() for ex in examples)

    def test_identifies_diagram_complexity(self):
        """Test identification of diagram complexity levels"""
        simple_diagram = "graph TD\n    A --> B\n    B --> C"
        complex_diagram = """
graph TB
    subgraph "Region 1"
        subgraph "VPC 1"
            LB1[Load Balancer]
            subgraph "AZ 1a"
                Web1[Web Tier]
                App1[App Tier]
            end
            subgraph "AZ 1b"
                Web2[Web Tier]
                App2[App Tier]
            end
        end
    end
    
    subgraph "Region 2"
        subgraph "VPC 2"
            LB2[Load Balancer]
            subgraph "AZ 2a"
                Web3[Web Tier]
                App3[App Tier]
            end
        end
    end
    
    subgraph "Global Services"
        CDN[CloudFront CDN]
        DNS[Route 53]
        DB[(Multi-Region Aurora)]
    end
    
    DNS --> CDN
    CDN --> LB1
    CDN --> LB2
    LB1 --> Web1
    LB1 --> Web2
    Web1 --> App1
    Web2 --> App2
    App1 --> DB
    App2 --> DB
"""
        
        assert self.collector._determine_diagram_complexity(simple_diagram) == "basic"
        assert self.collector._determine_diagram_complexity(complex_diagram) == "enterprise"

    def test_saves_corpus_to_jsonl(self):
        """Test corpus saving to JSONL format"""
        # Create sample examples
        examples = [
            {
                "prompt": "Create a Mermaid diagram showing microservices architecture",
                "completion": "```mermaid\ngraph TD\n    Gateway --> ServiceA\n    Gateway --> ServiceB\n```",
                "metadata": {
                    "source": "real_production_repository",
                    "category": "architecture_flowchart",
                    "authentic": True,
                    "language": "mermaid"
                }
            }
        ]
        
        # Save corpus
        output_file = "test_mermaid_corpus.jsonl"
        self.collector.save_corpus(examples, output_file)
        
        # Verify file was created and contains valid JSONL
        output_path = Path(self.temp_dir) / output_file
        assert output_path.exists()
        
        with open(output_path, 'r') as f:
            lines = f.readlines()
        
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["prompt"] == examples[0]["prompt"]
        assert parsed["metadata"]["authentic"] == True

    def test_collector_finds_minimum_examples(self):
        """Test collector can find minimum number of quality examples"""
        # This test will initially fail until collector is implemented
        examples = self.collector.collect_real_mermaid_examples(target_count=20)
        
        # Should collect at least 20 examples
        assert len(examples) >= 20
        
        # All examples should be valid
        for example in examples:
            assert "prompt" in example
            assert "completion" in example
            assert "metadata" in example
            assert example["metadata"]["authentic"] == True

    def test_excludes_template_placeholders(self):
        """Test collector excludes obvious template/placeholder content"""
        template_content = """
```mermaid
graph TD
    A[Your Component Here] --> B[Your Service]
    B --> C[Your Database]
    %% Add more components as needed
```
"""
        
        # Should reject template content
        assert self.collector._is_quality_mermaid_content(template_content) == False

if __name__ == "__main__":
    pytest.main([__file__, "-v"])