# Datus Agent Directory Structure Migration Guide

## Overview

This document summarizes the directory structure refactoring that occurred after commit `3bc9793`. The goal was to consolidate all Datus-related files under a unified `{agent.home}` directory (default: `~/.datus`) for better organization and management.

---

## Change History

This refactoring consolidated all Datus-related files under a unified `{agent.home}` directory (default: `~/.datus`) for better organization and management. The changes were implemented across 5 commits from October 2025.

### Core Changes

**Infrastructure & Path Management**
- ✅ Added `datus/utils/path_manager.py` - Unified path manager with centralized directory management
- ✅ Added `agent.home` configuration to `agent.yml` for customizable installation directory
- ✅ Implemented automatic directory creation with `ensure_dirs()` method

**Directory Migration**
- ✅ Configuration files: `conf/` → `{agent.home}/conf/`
- ✅ Logs: `logs/` → `{agent.home}/logs/`
- ✅ Sessions: `sessions/` → `{agent.home}/sessions/`
- ✅ Templates: `template/` → `{agent.home}/template/`
- ✅ Trajectories: `trajectory/` → `{agent.home}/trajectory/`
- ✅ Output results: `save/` → `{agent.home}/save/`
- ✅ Benchmark datasets: `benchmark/` → `{agent.home}/benchmark/`

**Sub-Agent Support**
- ✅ Semantic models: `{agent.home}/semantic_models/{namespace}/`
- ✅ SQL summaries: `{agent.home}/sql_summaries/{namespace}/`
- ✅ Sub-agent workspace: `{agent.home}/workspace/{sub_agent_name}/`
- ✅ Sub-agent data: `{agent.home}/data/sub_agents/{agent_name}/`

**Configuration Simplification**
- ✅ Removed `benchmark` configuration section from `agent.yml`
- ✅ Removed `--benchmark_path` CLI parameter
- ✅ Removed `--output_dir` CLI parameter (deprecated)
- ✅ Removed `--trajectory_dir` CLI parameter (deprecated)
- ✅ Added deprecation warnings to guide users to new approach
- ✅ Standardized benchmark subdirectory structure:
  - `bird_dev` → `{agent.home}/benchmark/bird/`
  - `spider2` → `{agent.home}/benchmark/spider2/`
  - `semantic_layer` → `{agent.home}/benchmark/semantic_layer/`

**Component Updates**
- ✅ Updated MCP manager, prompt manager, session manager to use new paths
- ✅ Updated agentic node implementations for semantic model and SQL summary generation
- ✅ Updated all CLI commands and documentation

### Impact Summary

- **Total Files Changed**: 55
- **Lines Added**: 885
- **Lines Deleted**: 448
- **Net Change**: +437 lines

---

## New Directory Structure

```
{agent.home}/  (default: ~/.datus/)
│
├── conf/                          # Configuration files
│   ├── agent.yml                  # Main configuration
│   ├── .mcp.json                  # MCP configuration
│   └── auth_clients.yml           # Authentication config
│
├── data/                          # Data storage
│   ├── datus_db_{namespace}/      # RAG storage (per namespace)
│   └── sub_agents/                # Sub-agent data
│       ├── gen_semantic_model/
│       ├── gen_metrics/
│       └── gen_sql_summary/
│
├── logs/                          # Log files
│   └── datus.log
│
├── sessions/                      # Session databases
│   └── {session_id}.db
│
├── template/                      # Template files
│
├── sample/                        # Sample files
│
├── run/                          # Runtime files
│   └── datus-agent-api.pid       # PID file
│
├── benchmark/                     # Benchmark datasets ⭐ New migration
│   ├── bird/                      # BIRD-DEV benchmark
│   ├── spider2/                   # Spider2 benchmark
│   └── semantic_layer/            # Semantic layer benchmark
│       └── success_story.csv      # Historical SQL cases
│
├── save/                          # Output results (previously migrated)
│   └── {namespace}/               # Organized by namespace
│
├── metricflow/                    # MetricFlow configuration
│   └── env_settings.yml
│
├── workspace/                     # Workspace (previously added)
│   └── {sub_agent_name}/
│
├── trajectory/                    # Execution trajectories (previously migrated)
│   └── {task_id}_{timestamp}.yaml
│
├── semantic_models/               # Semantic models (previously added)
│   └── {namespace}/
│
├── sql_summaries/                 # SQL summaries (previously added)
│   └── {namespace}/
│
└── history                        # Command history
```

---

## Key Changes

### 1. path_manager.py - Unified Path Management

```python
from datus.utils.path_manager import get_path_manager

pm = get_path_manager()

# Directory properties
pm.conf_dir          # {home}/conf
pm.data_dir          # {home}/data
pm.logs_dir          # {home}/logs
pm.sessions_dir      # {home}/sessions
pm.benchmark_dir     # {home}/benchmark ⭐ New
pm.save_dir          # {home}/save
pm.trajectory_dir    # {home}/trajectory
pm.workspace_dir     # {home}/workspace
pm.semantic_models_dir  # {home}/semantic_models
pm.sql_summaries_dir    # {home}/sql_summaries

# Configuration file paths
pm.agent_config_path()    # {home}/conf/agent.yml
pm.mcp_config_path()      # {home}/conf/.mcp.json
pm.metricflow_config_path()  # {home}/metricflow/env_settings.yml

# Data paths
pm.rag_storage_path(namespace)     # {home}/data/datus_db_{namespace}
pm.sub_agent_path(agent_name)      # {home}/data/sub_agents/{agent_name}
pm.session_db_path(session_id)     # {home}/sessions/{session_id}.db
pm.semantic_model_path(namespace)  # {home}/semantic_models/{namespace}
pm.sql_summary_path(namespace)     # {home}/sql_summaries/{namespace}
```

### 2. agent_config.py - Benchmark Path Management ⭐

**Old Way (Removed)**:
```yaml
benchmark:
  bird_dev:
    benchmark_path: benchmark/bird/dev_20240627  # ❌ No longer needed
  spider2:
    benchmark_path: benchmark/spider2/spider2-snow  # ❌ No longer needed
```

**New Way - Automatic Mapping**:
```python
agent_config.benchmark_path("bird_dev")
# Returns: {agent.home}/benchmark/bird/

agent_config.benchmark_path("spider2")
# Returns: {agent.home}/benchmark/spider2/

agent_config.benchmark_path("semantic_layer")
# Returns: {agent.home}/benchmark/semantic_layer/
```

### 3. Simplified CLI Parameters

**Old Way (Deprecated)**:
```bash
# ❌ Old way (deprecated)
datus bootstrap-kb \
  --benchmark bird_dev \
  --benchmark_path ~/my_custom_path/bird  # Removed

# ❌ Old way (deprecated)
datus ask \
  --output_dir ./my_output \               # Deprecated
  --trajectory_dir ./my_trajectories       # Deprecated
```

**New Way**:
```bash
# ✅ New way - Just configure agent.home
# In agent.yml:
agent:
  home: ~/.datus  # Or custom path

# Command line usage
datus benchmark --benchmark bird_dev --namespace bird_sqlite
datus bootstrap-kb --namespace your_namespace
```

---

## Key Benefits

1. **Unified Management**: All Datus files centralized in one configurable directory
2. **Simplified Configuration**: Removed many path configurations, automatic path derivation
3. **Easy Migration**: Just modify `agent.home` to migrate all data
4. **Clear Structure**: Clear directory responsibilities, easy to maintain and backup
5. **Backward Compatible**: Old parameters show deprecation warnings, guiding users to new approach

---

## Migration Guide

### Configuration File Update

**Before** (`agent.yml`):
```yaml
agent:
  # Multiple path configurations
  storage_path: data
  output_dir: save
  trajectory_dir: trajectory

  benchmark:
    bird_dev:
      benchmark_path: benchmark/bird/dev_20240627
    spider2:
      benchmark_path: benchmark/spider2/spider2-snow
```

**After** (`~/.datus/conf/agent.yml`):
```yaml
agent:
  home: ~/.datus  # Optional, defaults to ~/.datus

  # ❌ No longer need these configurations
  # All paths are automatically managed under {agent.home}
```

### Data Migration

If you have existing data, migrate to new structure:

```bash
# Assuming data was previously in project root
mv ./benchmark ~/.datus/
mv ./save ~/.datus/
mv ./data ~/.datus/
mv ./logs ~/.datus/
mv ./sessions ~/.datus/

# Or if you set a custom home
AGENT_HOME=/path/to/custom/home
mv ./benchmark $AGENT_HOME/
mv ./save $AGENT_HOME/
mv ./data $AGENT_HOME/
mv ./logs $AGENT_HOME/
mv ./sessions $AGENT_HOME/
```

### Command Line Updates

**Old Commands (Will show deprecation warnings)**:
```bash
datus ask \
  --output_dir ./custom_output \
  --trajectory_dir ./custom_trajectories \
  "What is the total sales?"

datus bootstrap-kb \
  --benchmark bird_dev \
  --benchmark_path ~/custom_benchmark_path
```

**New Commands**:
```bash
# Configure agent.home in agent.yml first, then:
datus ask "What is the total sales?"

datus bootstrap-kb --benchmark bird_dev
```

---

## Special Notes

### Success Story Files

Success story CSV files (historical SQL cases) are saved to:
- `{agent.home}/benchmark/{subagent_name}/success_story.csv`

This is used by the web UI's `save_success_story()` method and the `datus bootstrap-kb --success_story` command.

Default location for semantic_layer benchmark:
- `{agent.home}/benchmark/semantic_layer/success_story.csv`

### Backward Compatibility

The system maintains backward compatibility by:
1. Showing deprecation warnings for old parameters
2. Guiding users to use `agent.home` configuration
3. Automatically creating directories when they don't exist

---

## Summary

This refactoring series completely unified the Datus Agent directory structure, making configuration simpler and management more convenient. All Datus-related files are now organized under a single configurable directory, with clear responsibilities and automatic path management. 🎉
