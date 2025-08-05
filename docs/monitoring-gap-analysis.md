# Monitoring Gap Analysis - Data Preparation Phase

## Problem Statement

The current monitoring system incorrectly reports `TIMED_OUT` status during the data preparation phase because:

1. **No heartbeat updates**: The `prepare_mistral_dataset.py` script only updates S3 metadata at initialization
2. **Long-running downloads**: Policy data downloads can take 15-30+ minutes with no status updates
3. **False positives**: Monitoring system assumes process is dead when it's actually working normally

## Current Behavior

### What happens:
- Pipeline starts and posts initial metadata to S3 
- Data preparation begins downloading 8 policy sources (27GB+ of data)
- No metadata updates occur during downloads/processing
- Monitoring system detects stale heartbeat after 10 minutes
- Status incorrectly shows `TIMED_OUT` instead of `RUNNING`

### Example timeline:
```
19:27:00 - Pipeline starts, posts metadata
19:27:30 - Data prep begins downloading
19:28:00-19:40:00 - Downloads in progress, NO UPDATES
19:40:00 - Monitor detects TIMED_OUT (false positive)
19:45:00 - Data prep completes, moves to training
```

## Impact

- **Operational confusion**: Can't distinguish between stuck and working processes
- **False alerts**: May trigger unnecessary intervention
- **Loss of visibility**: No insight into data prep progress (which folder, how many files, etc.)

## Solution Requirements

1. **Heartbeat during downloads**: Update metadata every 2-5 minutes during downloads
2. **Progress indicators**: Show current phase, folder being processed, file counts
3. **Granular status**: Distinguish between `downloading`, `processing`, `extracting`, etc.
4. **Error handling**: Proper error states if downloads actually fail
5. **Backwards compatibility**: Don't break existing monitoring logic

## Proposed Architecture

### Enhanced Metadata Structure
```json
{
  "run_id": "mistral-20250805-142652",
  "status": "running",
  "phase": "data_prep",
  "sub_phase": "downloading_policy",
  "current_operation": "corpus_federal",
  "progress": {
    "policy_sources_completed": 3,
    "policy_sources_total": 8,
    "files_downloaded": 1250,
    "estimated_completion": "2025-08-05T20:15:00Z"
  },
  "last_update": "2025-08-05T19:35:00Z",
  "heartbeat_interval": 180
}
```

### Heartbeat Implementation
- Update S3 metadata every 3 minutes during long operations
- Include specific progress information
- Use background thread to avoid blocking main operations
- Fail gracefully if S3 updates fail

## Files to Modify

1. `scripts/mistral/prepare_mistral_dataset.py` - Add heartbeat system
2. `scripts/monitoring/core.py` - Handle new metadata structure  
3. `scripts/monitoring/monitor.py` - Display enhanced progress info
4. `scripts/mistral/run_mistral_training_pipeline.sh` - Update metadata calls

## Testing Strategy

1. **Unit tests**: Test heartbeat functionality in isolation
2. **Integration tests**: Run data prep with monitoring active
3. **Failure scenarios**: Test behavior when S3 updates fail
4. **Performance impact**: Ensure heartbeats don't slow downloads

## Timeline

- **Phase 1**: Document gap (✅ Current)
- **Phase 2**: Design heartbeat system  
- **Phase 3**: Implement in data prep script
- **Phase 4**: Update monitoring components
- **Phase 5**: Test and validate

## Success Criteria

- ✅ Monitoring shows accurate status during entire pipeline
- ✅ Progress visibility into data preparation phases  
- ✅ No false positive TIMED_OUT alerts
- ✅ Maintains existing monitoring API compatibility
- ✅ Minimal performance impact on data operations