---
name: etl-job-lifecycle
description: Manage ETL work as a verifiable lifecycle from SQL generation and DDL setup through job submission, run monitoring, output validation, and final pass or fail reporting
tags:
  - data-engineering
  - etl
  - orchestration
  - validation
  - workflow
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
---

# ETL Job Lifecycle

Use this skill when an ETL task must be handled as a full lifecycle instead of isolated SQL generation. This skill is for **submit, monitor, validate, and close out** workflows.

## When to use this skill

Activate when you need to:

- generate SQL and then submit a real ETL job
- connect build, run, and verification stages
- monitor job execution and summarize run results
- block promotion if the output fails validation

## Core workflow

1. Prepare or validate the target DDL.
2. Generate or review transformation SQL.
3. Submit the ETL job to the execution system.
4. Poll job or node status until completion.
5. Validate output tables with contract and quality checks.
6. Return a lifecycle report with build status, run status, output validation, and next action.

## Bundled resources

- For the recommended stage sequence, read [references/stage-sequence.md](references/stage-sequence.md).
- To start a job lifecycle manifest, copy [assets/etl_job_manifest.template.json](assets/etl_job_manifest.template.json).

## Output expectations

At minimum, return:

- job identifier
- submission status
- final run state
- output objects
- validation summary
- promotion or rollback decision

