#!/usr/bin/env python3

import json
import subprocess
import re
import time

def get_all_issues():
    """Get all issues with their content"""
    result = subprocess.run(
        ['gh', 'issue', 'list', '--limit', '50', '--json', 'number,title,body'],
        capture_output=True, text=True
    )
    return json.loads(result.stdout)

def parse_metadata_from_body(body):
    """Parse metadata from issue body"""
    metadata = {}
    
    # Priority
    priority_match = re.search(r'\*\*Priority\*\*:\s*(\w+)', body)
    if priority_match:
        priority = priority_match.group(1).lower()
        if priority == 'high':
            metadata['priority'] = 'priority:high'
        elif priority == 'medium':
            metadata['priority'] = 'priority:medium'
        # No label for low priority
    
    # Estimate
    estimate_match = re.search(r'\*\*Estimate\*\*:\s*(\d+)\s*hours?', body)
    if estimate_match:
        hours = int(estimate_match.group(1))
        if hours <= 1:
            metadata['estimate'] = 'estimate:1h'
        elif hours <= 2:
            metadata['estimate'] = 'estimate:2h'
        elif hours <= 4:
            metadata['estimate'] = 'estimate:4h'
        elif hours <= 8:
            metadata['estimate'] = 'estimate:8h'
        else:
            metadata['estimate'] = 'estimate:1d'
    
    # Size
    size_match = re.search(r'\*\*Size\*\*:\s*(\w+)', body)
    if size_match:
        size = size_match.group(1).lower()
        metadata['size'] = f'size:{size}'
    
    # Milestone
    milestone_match = re.search(r'\*\*Milestone\*\*:\s*M(\d+)', body)
    if milestone_match:
        milestone_num = milestone_match.group(1)
        metadata['milestone'] = f'milestone:m{milestone_num}'
    
    return metadata

def add_labels_to_issue(issue_number, labels):
    """Add labels to an issue"""
    if not labels:
        return
    
    label_args = []
    for label in labels:
        label_args.extend(['--add-label', label])
    
    cmd = ['gh', 'issue', 'edit', str(issue_number)] + label_args
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print(f"✓ Added labels to issue #{issue_number}: {', '.join(labels)}")
    else:
        print(f"✗ Failed to add labels to issue #{issue_number}: {result.stderr}")

def main():
    print("Fetching all issues...")
    issues = get_all_issues()
    
    print(f"Found {len(issues)} issues. Processing...")
    
    for issue in issues:
        number = issue['number']
        body = issue['body'] or ''
        title = issue['title']
        
        print(f"\nProcessing issue #{number}: {title}")
        
        metadata = parse_metadata_from_body(body)
        
        if metadata:
            labels_to_add = list(metadata.values())
            add_labels_to_issue(number, labels_to_add)
            time.sleep(0.5)  # Rate limiting
        else:
            print(f"  No metadata found in issue #{number}")

if __name__ == "__main__":
    main()