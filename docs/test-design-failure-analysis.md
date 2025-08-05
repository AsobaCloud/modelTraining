# Test Design Failure Analysis

## Problem Statement

We repeatedly encounter "edge case" failures that are actually **predictable, core failure modes** that should have been covered by basic testing. This indicates a fundamental lack of test design methodology.

## Pattern of Failures

### Historical "Surprises" That Should Have Been Tested

1. **Data validation fails** → Pipeline stops silently without updating monitoring
2. **Long downloads timeout** → Monitoring shows TIMED_OUT instead of IN_PROGRESS  
3. **S3 permissions errors** → No graceful handling or clear error messages
4. **SSH connection drops** → Instance operations fail with cryptic errors
5. **Disk space exhaustion** → No pre-flight checks or graceful degradation
6. **Network interruptions** → Downloads fail without retry logic
7. **Malformed data sources** → Processing crashes without error context

### The Real Issue

These aren't "edge cases" - they're **core operational scenarios** that happen regularly in production systems. The fact that we keep being surprised by them indicates:

- **No failure mode analysis** during design
- **No negative test cases** in development  
- **No chaos engineering** or fault injection
- **No operational runbooks** for common failures

## Required Test Design Framework

### 1. Failure Mode and Effects Analysis (FMEA)

For every component, systematically analyze:

| Component | Failure Mode | Effect | Detection | Mitigation | Test Case |
|-----------|--------------|--------|-----------|------------|-----------|
| Data Download | S3 timeout | Pipeline stalls | Heartbeat stops | Retry with backoff | `test_s3_timeout()` |
| Data Download | Invalid credentials | Access denied | AWS error | Clear error message | `test_invalid_credentials()` |
| Data Validation | Missing required field | Validation fails | Schema check | Skip record + log | `test_missing_text_field()` |
| Data Validation | Malformed JSON | Parse error | JSON exception | Graceful skip | `test_malformed_json()` |
| SSH Operations | Network disconnect | Command fails | SSH timeout | Retry connection | `test_ssh_disconnect()` |
| Monitoring | S3 write fails | Status not updated | S3 error | Local fallback | `test_s3_monitoring_failure()` |
| Disk Space | EBS full | Write operations fail | df check | Pre-flight validation | `test_disk_space_exhaustion()` |

### 2. Chaos Engineering Test Suite

**Network Failures**
```bash
# Simulate packet loss during downloads
test_network_packet_loss() {
    tc qdisc add dev eth0 root netem loss 10%
    run_data_prep_and_verify_retry_behavior
}

# Simulate complete network outage
test_network_outage() {
    iptables -A OUTPUT -j DROP
    run_data_prep_and_verify_graceful_failure
}
```

**Resource Exhaustion**
```bash
# Fill up disk space during processing
test_disk_full() {
    dd if=/dev/zero of=/mnt/training/fill_disk bs=1M count=1000
    run_data_prep_and_verify_error_handling
}

# Consume all available memory
test_memory_exhaustion() {
    stress-ng --vm 1 --vm-bytes 90% --timeout 60s &
    run_data_prep_and_verify_graceful_degradation
}
```

**Service Failures**
```bash
# S3 returns 503 errors
test_s3_service_unavailable() {
    aws_mock_return_503_for_bucket "policy-database"
    run_data_prep_and_verify_retry_with_backoff
}

# EC2 instance gets terminated mid-process
test_instance_termination() {
    run_data_prep_in_background
    sleep 30 && terminate_instance
    verify_monitoring_shows_instance_terminated
}
```

### 3. Data Quality Test Cases

**Schema Validation**
```python
def test_missing_required_fields():
    """Test handling of records missing 'text' field"""
    malformed_data = [
        {"source": "test.pdf"},  # Missing 'text'
        {"text": "", "source": "test2.pdf"},  # Empty 'text'  
        {"text": None, "source": "test3.pdf"},  # Null 'text'
    ]
    result = process_data(malformed_data)
    assert result.errors_logged == 3
    assert result.valid_records == 0
    assert monitoring_shows_validation_errors()

def test_data_size_limits():
    """Test handling of extremely large records"""
    huge_text = "x" * (10 * 1024 * 1024)  # 10MB text field
    record = {"text": huge_text, "source": "huge.pdf"}
    result = process_data([record])
    assert result.truncated_records == 1
    assert len(result.processed_records[0]["text"]) < 1024 * 1024

def test_encoding_issues():
    """Test handling of various text encodings"""
    test_cases = [
        {"text": "café", "encoding": "utf-8"},
        {"text": "测试", "encoding": "utf-8"},  
        {"text": b'\xff\xfe\x00\x00', "encoding": "invalid"},
    ]
    result = process_data_with_encodings(test_cases)
    assert result.encoding_errors_handled == 1
```

### 4. Integration Test Matrix

| Scenario | Expected Behavior | Monitoring Status | Alert Behavior |
|----------|-------------------|-------------------|----------------|
| Happy path completion | Datasets uploaded to S3 | `COMPLETED` | Success notification |
| S3 download timeout | Retry with exponential backoff | `RETRYING` | No alert (normal retry) |
| Repeated S3 failures | Fail after max retries | `FAILED` | Critical alert with context |
| Validation errors | Skip bad records, continue | `RUNNING` with warnings | Warning notification |
| All records invalid | Fail with clear message | `FAILED` | Critical alert: No valid data |
| Instance terminated | Mark as infrastructure failure | `INSTANCE_TERMINATED` | Infrastructure alert |
| Disk space exhaustion | Fail with clear resource error | `FAILED` | Resource exhaustion alert |

### 5. Monitoring Test Requirements

**Test every monitoring state transition:**
```python
def test_monitoring_state_transitions():
    """Verify monitoring captures all state changes"""
    
    # Test normal progression
    states = run_pipeline_and_capture_states()
    expected = ["INITIALIZING", "DOWNLOADING", "PROCESSING", "COMPLETED"]
    assert states == expected
    
    # Test failure states  
    inject_validation_failure()
    states = run_pipeline_and_capture_states()
    assert states[-1] == "FAILED"
    assert get_failure_reason() == "Validation failed: Missing text fields"
    
    # Test timeout detection
    block_heartbeat_updates()
    time.sleep(TIMEOUT_THRESHOLD + 1)
    assert monitor_detects_timeout()
```

**Test alert delivery:**
```python
def test_alert_delivery():
    """Verify alerts are sent for all failure scenarios"""
    
    for failure_type in FAILURE_SCENARIOS:
        inject_failure(failure_type)
        run_pipeline()
        
        alert = get_latest_slack_alert()
        assert alert is not None
        assert failure_type in alert.message
        assert alert.contains_actionable_guidance()
        assert alert.contains_instance_details()
```

## Implementation Requirements

### 1. Test Infrastructure Setup
- **Mock services** for S3, SSH, networking
- **Fault injection** capabilities  
- **Resource limitation** controls
- **Monitoring validation** harness

### 2. Automated Test Execution
- Run failure tests in **every CI build**
- **Pre-deployment validation** with full failure suite
- **Nightly chaos engineering** runs in staging

### 3. Test Documentation
- **Runbook** for each failure scenario
- **Expected behavior** documentation
- **Recovery procedures** for each failure type

## Success Criteria

✅ **Zero "surprising" failures** in production  
✅ **All failure modes** have corresponding test cases  
✅ **Monitoring accuracy** at 100% (no false positives/negatives)  
✅ **Mean time to diagnosis** under 2 minutes for any failure  
✅ **Automated recovery** for all transient failures  

## Next Steps

1. **Catalog all possible failure modes** for current pipeline
2. **Implement test harness** for fault injection  
3. **Create test cases** for every failure mode
4. **Run full test suite** and fix all gaps
5. **Integrate into CI/CD** pipeline
6. **Document operational procedures** for each failure type

The goal is to **never be surprised** by a system failure again. Every failure should be a known scenario with a tested response.