#!/usr/bin/env python3
"""
Open Source Projects Shell Script Collector
Collects shell scripts from popular infrastructure and DevOps projects

Priority 2 Source: Target 125 examples from popular open source projects
"""

import os
import json
import requests
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Optional
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OpenSourceCollector:
    """Collects shell scripts from popular open source projects"""
    
    def __init__(self, output_dir: str = "opensource_corpus"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Popular infrastructure projects with known shell scripts
        self.target_projects = {
            # Infrastructure Tools
            'docker/docker-ce': {'type': 'container', 'scripts': ['**/install.sh', '**/setup.sh']},
            'kubernetes/kubernetes': {'type': 'orchestration', 'scripts': ['**/*deploy*.sh', '**/setup*.sh']},
            'hashicorp/terraform': {'type': 'infrastructure', 'scripts': ['**/install*.sh', '**/build*.sh']},
            'helm/helm': {'type': 'package_manager', 'scripts': ['**/install*.sh', 'scripts/*.sh']},
            
            # Monitoring & Observability
            'prometheus/prometheus': {'type': 'monitoring', 'scripts': ['**/install*.sh', 'scripts/*.sh']},
            'grafana/grafana': {'type': 'monitoring', 'scripts': ['**/setup*.sh', 'scripts/*.sh']},
            'jaegertracing/jaeger': {'type': 'monitoring', 'scripts': ['**/deploy*.sh', 'scripts/*.sh']},
            
            # Service Mesh & Networking
            'istio/istio': {'type': 'service_mesh', 'scripts': ['**/install*.sh', 'tools/*.sh']},
            'envoyproxy/envoy': {'type': 'service_mesh', 'scripts': ['**/build*.sh', 'tools/*.sh']},
            
            # CI/CD & GitOps
            'argoproj/argo-cd': {'type': 'gitops', 'scripts': ['**/install*.sh', 'hack/*.sh']},
            'fluxcd/flux2': {'type': 'gitops', 'scripts': ['**/install*.sh', 'scripts/*.sh']},
            'tektoncd/pipeline': {'type': 'cicd', 'scripts': ['**/install*.sh', 'scripts/*.sh']},
            
            # Cloud Native Storage
            'rook/rook': {'type': 'storage', 'scripts': ['**/deploy*.sh', 'scripts/*.sh']},
            'longhorn/longhorn': {'type': 'storage', 'scripts': ['**/install*.sh', 'scripts/*.sh']},
            
            # Security
            'falcosecurity/falco': {'type': 'security', 'scripts': ['**/install*.sh', 'scripts/*.sh']},
            'aquasecurity/trivy': {'type': 'security', 'scripts': ['**/install*.sh', 'scripts/*.sh']},
        }
        
        self.collected_examples = []
        
    def collect_opensource_examples(self) -> List[Dict]:
        """Main collection method for open source projects"""
        logger.info(f"Starting collection from {len(self.target_projects)} open source projects")
        
        all_examples = []
        
        for project, config in self.target_projects.items():
            logger.info(f"Processing project: {project}")
            try:
                project_examples = self._collect_project_examples(project, config)
                all_examples.extend(project_examples)
                logger.info(f"Collected {len(project_examples)} examples from {project}")
            except Exception as e:
                logger.warning(f"Failed to collect from {project}: {e}")
        
        # Add curated examples for known patterns
        curated_examples = self._get_curated_examples()
        all_examples.extend(curated_examples)
        
        logger.info(f"Total collected: {len(all_examples)} examples")
        return all_examples
    
    def _collect_project_examples(self, project: str, config: Dict) -> List[Dict]:
        """Collect examples from a specific project"""
        examples = []
        project_type = config['type']
        
        # Use GitHub API to get file contents without cloning
        try:
            api_examples = self._get_files_via_api(project, config['scripts'], project_type)
            examples.extend(api_examples)
        except Exception as e:
            logger.debug(f"API collection failed for {project}: {e}")
        
        # If API fails, try known patterns for the project type
        if not examples:
            pattern_examples = self._get_pattern_examples(project, project_type)
            examples.extend(pattern_examples)
        
        return examples
    
    def _get_files_via_api(self, project: str, script_patterns: List[str], project_type: str) -> List[Dict]:
        """Get shell scripts via GitHub API"""
        examples = []
        
        # Search for shell files in the repository
        search_url = f"https://api.github.com/search/code"
        
        for pattern in ['install.sh', 'setup.sh', 'deploy.sh', 'build.sh']:
            params = {
                'q': f'filename:{pattern} repo:{project}',
                'per_page': 5  # Limit to avoid rate limits
            }
            
            try:
                response = requests.get(search_url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    
                    for item in data.get('items', []):
                        file_examples = self._process_github_file(item, project, project_type)
                        examples.extend(file_examples)
                        
                elif response.status_code == 403:
                    # Rate limit hit, break
                    logger.warning(f"GitHub API rate limit hit for {project}")
                    break
                    
            except Exception as e:
                logger.debug(f"Failed to search {pattern} in {project}: {e}")
        
        return examples[:10]  # Limit per project
    
    def _process_github_file(self, file_item: Dict, project: str, project_type: str) -> List[Dict]:
        """Process a GitHub file and extract training examples"""
        examples = []
        
        try:
            # Get file content
            download_url = file_item.get('download_url')
            if not download_url:
                return examples
            
            response = requests.get(download_url, timeout=10)
            if response.status_code != 200:
                return examples
            
            content = response.text
            
            # Skip very large files
            if len(content) > 10000:
                return examples
            
            # Generate training example
            file_path = file_item.get('path', '')
            filename = os.path.basename(file_path)
            
            prompt = self._generate_prompt_for_opensource_script(filename, project_type, project)
            clean_content = self._clean_script_content(content)
            
            example = {
                "prompt": prompt,
                "completion": f"```bash\n{clean_content}\n```",
                "metadata": {
                    "source": "opensource_project",
                    "project": project,
                    "project_type": project_type,
                    "filename": filename,
                    "file_path": file_path
                }
            }
            
            examples.append(example)
            
        except Exception as e:
            logger.debug(f"Failed to process file from {project}: {e}")
        
        return examples
    
    def _get_pattern_examples(self, project: str, project_type: str) -> List[Dict]:
        """Generate examples based on known patterns for project types"""
        examples = []
        
        patterns = {
            'container': {
                'docker': [
                    {
                        'prompt': 'Write a shell script to install Docker on Ubuntu',
                        'script': '''#!/bin/bash
# Install Docker on Ubuntu

set -e

# Update package index
sudo apt-get update

# Install dependencies
sudo apt-get install -y \\
    apt-transport-https \\
    ca-certificates \\
    curl \\
    gnupg \\
    lsb-release

# Add Docker GPG key
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg

# Add Docker repository
echo \\
  "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu \\
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Update package index again
sudo apt-get update

# Install Docker
sudo apt-get install -y docker-ce docker-ce-cli containerd.io

# Start and enable Docker
sudo systemctl start docker
sudo systemctl enable docker

# Add current user to docker group
sudo usermod -aG docker $USER

echo "Docker installation completed. Please log out and back in to use Docker without sudo."'''
                    }
                ]
            },
            'orchestration': {
                'kubernetes': [
                    {
                        'prompt': 'Create a shell script to deploy Kubernetes cluster with kubeadm',
                        'script': '''#!/bin/bash
# Deploy Kubernetes cluster with kubeadm

set -e

NODE_TYPE="${1:-master}"

# Install dependencies
sudo apt-get update
sudo apt-get install -y apt-transport-https ca-certificates curl

# Add Kubernetes GPG key
curl -s https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -

# Add Kubernetes repository
echo "deb https://apt.kubernetes.io/ kubernetes-xenial main" | sudo tee /etc/apt/sources.list.d/kubernetes.list

# Install kubelet, kubeadm, kubectl
sudo apt-get update
sudo apt-get install -y kubelet kubeadm kubectl
sudo apt-mark hold kubelet kubeadm kubectl

if [ "$NODE_TYPE" = "master" ]; then
    # Initialize master node
    sudo kubeadm init --pod-network-cidr=10.244.0.0/16
    
    # Setup kubectl for current user
    mkdir -p $HOME/.kube
    sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
    sudo chown $(id -u):$(id -g) $HOME/.kube/config
    
    # Install Flannel network plugin
    kubectl apply -f https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml
    
    echo "Kubernetes master node setup completed"
else
    echo "Worker node ready. Run kubeadm join command from master node."
fi'''
                    }
                ]
            },
            'monitoring': {
                'prometheus': [
                    {
                        'prompt': 'Write a shell script to deploy Prometheus monitoring stack',
                        'script': '''#!/bin/bash
# Deploy Prometheus monitoring stack

set -e

NAMESPACE="${1:-monitoring}"

# Create monitoring namespace
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Add Prometheus Helm repository
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

# Install Prometheus stack
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \\
    --namespace $NAMESPACE \\
    --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.storageClassName=gp2 \\
    --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=20Gi \\
    --set grafana.persistence.enabled=true \\
    --set grafana.persistence.storageClassName=gp2 \\
    --set grafana.persistence.size=10Gi

# Wait for deployment
kubectl rollout status deployment/prometheus-grafana -n $NAMESPACE
kubectl rollout status statefulset/prometheus-prometheus-kube-prometheus-prometheus -n $NAMESPACE

# Get Grafana admin password
echo "Grafana admin password:"
kubectl get secret prometheus-grafana -n $NAMESPACE -o jsonpath="{.data.admin-password}" | base64 --decode
echo

echo "Prometheus monitoring stack deployed successfully in namespace: $NAMESPACE"'''
                    }
                ]
            }
        }
        
        project_patterns = patterns.get(project_type, {})
        
        for tool, tool_patterns in project_patterns.items():
            for pattern in tool_patterns:
                example = {
                    "prompt": pattern['prompt'],
                    "completion": f"```bash\n{pattern['script']}\n```",
                    "metadata": {
                        "source": "opensource_pattern",
                        "project": project,
                        "project_type": project_type,
                        "tool": tool
                    }
                }
                examples.append(example)
        
        return examples
    
    def _get_curated_examples(self) -> List[Dict]:
        """Get curated examples for common infrastructure patterns"""
        examples = [
            {
                "prompt": "Create a shell script to setup Kubernetes monitoring with Prometheus and Grafana",
                "completion": """```bash
#!/bin/bash
# Setup Kubernetes monitoring with Prometheus and Grafana

set -e

NAMESPACE="monitoring"

echo "Setting up Kubernetes monitoring stack..."

# Create monitoring namespace
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Add Helm repositories
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

# Install Prometheus
echo "Installing Prometheus..."
helm upgrade --install prometheus prometheus-community/prometheus \\
    --namespace $NAMESPACE \\
    --set server.persistentVolume.size=20Gi \\
    --set alertmanager.persistentVolume.size=5Gi

# Install Grafana
echo "Installing Grafana..."
helm upgrade --install grafana grafana/grafana \\
    --namespace $NAMESPACE \\
    --set persistence.enabled=true \\
    --set persistence.size=10Gi \\
    --set adminPassword=admin123

# Wait for deployments
kubectl rollout status deployment/prometheus-server -n $NAMESPACE
kubectl rollout status deployment/grafana -n $NAMESPACE

# Port forward for access
echo "Monitoring stack deployed. Access via:"
echo "Prometheus: kubectl port-forward -n $NAMESPACE svc/prometheus-server 9090:80"
echo "Grafana: kubectl port-forward -n $NAMESPACE svc/grafana 3000:80"
```""",
                "metadata": {
                    "source": "curated_example",
                    "category": "monitoring",
                    "type": "kubernetes_monitoring"
                }
            },
            {
                "prompt": "Write a shell script to deploy microservices with Istio service mesh",
                "completion": """```bash
#!/bin/bash
# Deploy microservices with Istio service mesh

set -e

NAMESPACE="${1:-default}"

echo "Deploying microservices with Istio..."

# Install Istio
echo "Installing Istio..."
curl -L https://istio.io/downloadIstio | sh -
export PATH=$PWD/istio-*/bin:$PATH

# Install Istio on cluster
istioctl install --set values.defaultRevision=default -y

# Enable sidecar injection
kubectl label namespace $NAMESPACE istio-injection=enabled --overwrite

# Deploy sample microservices
echo "Deploying sample application..."
kubectl apply -f - <<EOF
apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
  namespace: $NAMESPACE
spec:
  replicas: 2
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      containers:
      - name: frontend
        image: nginx:latest
        ports:
        - containerPort: 80
---
apiVersion: v1
kind: Service
metadata:
  name: frontend
  namespace: $NAMESPACE
spec:
  selector:
    app: frontend
  ports:
  - port: 80
    targetPort: 80
EOF

# Create Istio Gateway
kubectl apply -f - <<EOF
apiVersion: networking.istio.io/v1alpha3
kind: Gateway
metadata:
  name: frontend-gateway
  namespace: $NAMESPACE
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - "*"
---
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: frontend
  namespace: $NAMESPACE
spec:
  hosts:
  - "*"
  gateways:
  - frontend-gateway
  http:
  - route:
    - destination:
        host: frontend
        port:
          number: 80
EOF

echo "Microservices deployed with Istio service mesh"
echo "Access via: kubectl get svc istio-ingressgateway -n istio-system"
```""",
                "metadata": {
                    "source": "curated_example",
                    "category": "service_mesh",
                    "type": "istio_deployment"
                }
            },
            {
                "prompt": "Create a shell script for automated GitOps deployment with ArgoCD",
                "completion": """```bash
#!/bin/bash
# Automated GitOps deployment with ArgoCD

set -e

NAMESPACE="argocd"
REPO_URL="${1:-https://github.com/example/k8s-manifests}"
APP_NAME="${2:-sample-app}"

echo "Setting up GitOps with ArgoCD..."

# Create ArgoCD namespace
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Install ArgoCD
echo "Installing ArgoCD..."
kubectl apply -n $NAMESPACE -f https://raw.githubusercontent.com/argoproj/argo-cd/stable/manifests/install.yaml

# Wait for ArgoCD to be ready
echo "Waiting for ArgoCD to be ready..."
kubectl rollout status deployment/argocd-server -n $NAMESPACE

# Get ArgoCD admin password
ARGOCD_PASSWORD=$(kubectl -n $NAMESPACE get secret argocd-initial-admin-secret -o jsonpath="{.data.password}" | base64 -d)

# Install ArgoCD CLI
curl -sSL -o /tmp/argocd https://github.com/argoproj/argo-cd/releases/latest/download/argocd-linux-amd64
chmod +x /tmp/argocd
sudo mv /tmp/argocd /usr/local/bin/argocd

# Port forward ArgoCD server
kubectl port-forward svc/argocd-server -n $NAMESPACE 8080:443 &
PORTFORWARD_PID=$!

# Wait for port forward
sleep 5

# Login to ArgoCD
echo "Logging into ArgoCD..."
argocd login localhost:8080 --username admin --password $ARGOCD_PASSWORD --insecure

# Create application
echo "Creating ArgoCD application..."
argocd app create $APP_NAME \\
    --repo $REPO_URL \\
    --path manifests \\
    --dest-server https://kubernetes.default.svc \\
    --dest-namespace default \\
    --sync-policy automated \\
    --auto-prune \\
    --self-heal

# Sync application
argocd app sync $APP_NAME

# Kill port forward
kill $PORTFORWARD_PID

echo "GitOps deployment with ArgoCD completed"
echo "ArgoCD admin password: $ARGOCD_PASSWORD"
echo "Access ArgoCD: kubectl port-forward svc/argocd-server -n $NAMESPACE 8080:443"
```""",
                "metadata": {
                    "source": "curated_example",
                    "category": "gitops",
                    "type": "argocd_setup"
                }
            }
        ]
        
        return examples
    
    def _generate_prompt_for_opensource_script(self, filename: str, project_type: str, project: str) -> str:
        """Generate appropriate prompt for open source script"""
        
        # Filename-based prompts
        if 'install' in filename.lower():
            return f"Write a shell script to install {project.split('/')[-1]} on Linux"
        elif 'setup' in filename.lower():
            return f"Create a shell script to setup {project.split('/')[-1]} environment"
        elif 'deploy' in filename.lower():
            return f"Write a shell script to deploy {project.split('/')[-1]} infrastructure"
        elif 'build' in filename.lower():
            return f"Create a shell script to build {project.split('/')[-1]} from source"
        
        # Project type-based prompts
        type_prompts = {
            'container': f"Write a shell script for container deployment and management",
            'orchestration': f"Create a shell script for Kubernetes cluster setup",
            'infrastructure': f"Write a shell script for infrastructure provisioning",
            'monitoring': f"Create a shell script for monitoring stack deployment",
            'service_mesh': f"Write a shell script for service mesh configuration",
            'gitops': f"Create a shell script for GitOps workflow setup",
            'cicd': f"Write a shell script for CI/CD pipeline deployment",
            'storage': f"Create a shell script for persistent storage setup",
            'security': f"Write a shell script for security scanning and compliance"
        }
        
        return type_prompts.get(project_type, f"Write a shell script for {project_type} automation")
    
    def _clean_script_content(self, content: str) -> str:
        """Clean and sanitize script content"""
        import re
        
        # Remove very long lines (likely not real commands)
        lines = content.split('\n')
        clean_lines = []
        
        for line in lines:
            if len(line) < 200:  # Skip very long lines
                clean_lines.append(line)
        
        content = '\n'.join(clean_lines)
        
        # Replace sensitive patterns
        content = re.sub(r'\b\d{12}\b', '<ACCOUNT_ID>', content)
        content = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', '<EMAIL>', content)
        content = re.sub(r'https?://(?!.*\.(amazonaws\.com|k8s\.io|docker\.(com|io)|github\.com))[^\s"\']+', '<URL>', content)
        
        return content.strip()
    
    def save_corpus(self, examples: List[Dict], filename: str = "opensource_corpus.jsonl") -> None:
        """Save open source corpus to JSONL format"""
        output_file = self.output_dir / filename
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(examples)} open source examples to {output_file}")


def main():
    """Main open source collection pipeline"""
    logger.info("Starting Open Source Projects Collection")
    
    collector = OpenSourceCollector()
    
    try:
        # Collect examples
        examples = collector.collect_opensource_examples()
        
        # Save corpus
        collector.save_corpus(examples)
        
        # Generate statistics
        project_types = {}
        sources = {}
        
        for example in examples:
            metadata = example.get('metadata', {})
            ptype = metadata.get('project_type', 'unknown')
            source = metadata.get('source', 'unknown')
            
            project_types[ptype] = project_types.get(ptype, 0) + 1
            sources[source] = sources.get(source, 0) + 1
        
        logger.info(f"Open Source Collection Statistics:")
        logger.info(f"Total examples: {len(examples)}")
        logger.info(f"Project types: {dict(sorted(project_types.items()))}")
        logger.info(f"Sources: {dict(sorted(sources.items()))}")
        
        logger.info("Open source examples collection completed successfully!")
        
    except Exception as e:
        logger.error(f"Open source collection failed: {e}")
        raise


if __name__ == "__main__":
    main()