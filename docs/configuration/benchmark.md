# Benchmark

> Configure benchmark datasets and evaluation settings for Datus Agent

## Overview

Benchmark datasets are used to evaluate and test Datus Agent's performance on standardized SQL generation tasks. Benchmarks help measure accuracy, compare different configurations, and validate improvements.

## Supported Benchmarks

Datus Agent currently supports the following benchmark datasets:

* **bird_dev**: A comprehensive benchmark for complex SQL generation (BIRD-DEV)
* **spider2**: Advanced multi-database SQL benchmark
* **semantic_layer**: Business metric and semantic understanding benchmark

## Benchmark Directory Structure

Benchmark data is automatically stored at `{agent.home}/benchmark/{name}`:

```
{agent.home}/benchmark/
├── bird/              # BIRD-DEV benchmark data
├── spider2/           # Spider2 benchmark data
└── semantic_layer/    # Semantic layer benchmark data
```

**Note**: No configuration is required in `agent.yml`. The paths are automatically managed based on your `agent.home` setting.

## BIRD-DEV Benchmark

The BIRD (Big Bench for Large-scale Database Grounded Text-to-SQL Evaluation) benchmark tests complex SQL generation capabilities.

### Directory Location

Data should be placed at: `{agent.home}/benchmark/bird/`

### Usage

```bash
datus-agent benchmark --benchmark bird_dev --namespace bird_sqlite
```

## Spider2 Benchmark

Spider2 is an advanced benchmark that tests SQL generation across multiple databases and complex scenarios.

### Directory Location

Data should be placed at: `{agent.home}/benchmark/spider2/`

### Usage

```bash
datus-agent benchmark --benchmark spider2 --namespace snowflake
```

## Semantic Layer Benchmark

Tests the agent's ability to understand business metrics and semantic relationships.

### Directory Location

Data should be placed at: `{agent.home}/benchmark/semantic_layer/`

### Usage

```bash
datus-agent benchmark --benchmark semantic_layer --namespace your_namespace
```

For detailed usage instructions and advanced configuration options, see the [Benchmarks](/benchmarks) chapter.
