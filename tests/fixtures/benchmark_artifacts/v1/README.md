# Benchmark artifact schema fixture

`task-output.schema.json` is a test snapshot of the consumer-owned v1 schema
from `Datus-ai/datus-benchmark`, introduced by issue #139. Runtime code does
not load this copy; producer tests use it to detect contract drift without a
network or sibling-checkout dependency.
