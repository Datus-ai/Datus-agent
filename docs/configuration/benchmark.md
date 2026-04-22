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
  custom_bird:                       # Custom benchmark datasource
    benchmark_path: benchmark/custom_bird/dev_data

  custom_spider:
    benchmark_path: path/to/spider/data
```

## BIRD-DEV Benchmark

The BIRD (Big Bench for Large-scale Database Grounded Text-to-SQL Evaluation) benchmark tests complex SQL generation capabilities.

### Pre-configured Built-in Benchmarks

The `bird_dev`, `spider2`, and `semantic_layer` benchmarks are built-in and their paths are pre-configured in the system. You do not need (and cannot) override their `benchmark_path` in `agent.yml`.

For detailed usage instructions and advanced custom configuration options, see the [Benchmarks](../benchmark/benchmark_manual.md) chapter.