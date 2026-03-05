# Benchmark Configuration

Configure benchmark datasets to evaluate and test Datus Agent's performance on standardized SQL generation tasks. Benchmarks help measure accuracy, compare different configurations, and validate improvements.

## Supported Benchmarks

Datus Agent currently supports the following benchmark datasets:

- **BIRD-DEV**: A comprehensive benchmark for complex SQL generation
- **Spider2**: Advanced multi-database SQL benchmark
- **Semantic Layer**: Business metric and semantic understanding benchmark

## Benchmark Configuration Structure

Configure benchmarks in the `benchmark` section of your configuration file:

```yaml
benchmark:
  bird_dev:                          # Benchmark namespace
    benchmark_path: benchmark/bird/dev_20240627
    
  spider2:
    benchmark_path: benchmark/spider2/spider2-snow
    
  semantic_layer:
    benchmark_path: benchmark/semantic_layer
```

## BIRD-DEV Benchmark

The BIRD (Big Bench for Large-scale Database Grounded Text-to-SQL Evaluation) benchmark tests complex SQL generation capabilities.

### Configuration

```yaml
benchmark:
  bird_dev:
    benchmark_path: benchmark/bird/dev_20240627
```

## Spider2 Benchmark

Spider2 is an advanced benchmark that tests SQL generation across multiple databases and complex scenarios.

### Configuration

```yaml
benchmark:
  spider2:
    benchmark_path: benchmark/spider2/spider2-snow
```

## Semantic Layer Benchmark

Tests the agent's ability to understand business metrics and semantic relationships.

### Configuration

```yaml
benchmark:
  semantic_layer:
    benchmark_path: benchmark/semantic_layer
```

For detailed usage instructions and advanced configuration options, see the [Benchmarks](../benchmark/benchmark_manual.md) chapter.