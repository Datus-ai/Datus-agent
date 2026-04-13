---
name: pipeline-stage-orchestrator
description: Orchestrate dependent pipeline stages as verifiable gates where each stage executes, verifies, blocks on failure, and only advances when its predecessor has passed
tags:
  - data-engineering
  - orchestration
  - pipeline
  - validation
  - workflow
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
---

# Pipeline Stage Orchestrator

Use this skill when a multi-stage data workflow must be run as ordered, verifiable gates instead of loosely connected tasks. This skill is for **stage orchestration with pass or fail semantics**, not for decomposing a task across independent agents.

## When to use this skill

Activate when you need to:

- run build, write, ETL, and publish steps in order
- attach verification to every stage
- stop the pipeline immediately on a failed gate
- produce a concise stage-by-stage audit trail

## Core workflow

1. Define the ordered stages and dependencies.
2. For each stage, define:
   - execute action
   - verify action
   - rollback or halt action on failure
3. Run stages in order.
4. Mark a stage done only after verification passes.
5. Return a compact stage ledger with pass / fail and blocking reason.

## Bundled resources

- For the verifiable stage model, read [references/stage-model.md](references/stage-model.md).
- To start a stage manifest, copy [assets/stage_manifest.template.json](assets/stage_manifest.template.json).

## Output expectations

At minimum, return:

- ordered stages
- execution status per stage
- verification status per stage
- blocking stage if any
- final pipeline state

