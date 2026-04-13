---
name: bi-publish-verify
description: Publish BI dataset or dashboard changes with a verification-first workflow that compares refreshed outputs against expected key metrics before the change is considered complete
tags:
  - data-engineering
  - bi
  - publish
  - validation
  - metrics
version: "1.0.0"
user_invocable: false
disable_model_invocation: false
---

# BI Publish & Verify

Use this skill when BI SQL, semantic definitions, or dashboard wiring must be published and then verified against expected business metrics. This skill is for **post-publish validation**, not warehouse-side model generation.

## When to use this skill

Activate when you need to:

- update dataset SQL or semantic definitions
- publish a dashboard or BI dataset change
- compare refreshed BI metrics against expected values or ranges
- block rollout if the published result diverges materially

## Core workflow

1. Identify the dataset, dashboard, or semantic object being changed.
2. Capture the expected key metrics and tolerance.
3. Publish the BI change.
4. Refresh or query the published object.
5. Compare observed metrics against expectations.
6. Return a publish status plus metric diff report.

## Bundled resources

- For the publish verification checklist, read [references/publish-checklist.md](references/publish-checklist.md).
- To capture a publish contract, copy [assets/bi_publish_contract.template.json](assets/bi_publish_contract.template.json).

## Output expectations

At minimum, return:

- target BI object
- publish status
- refreshed metric values
- expected values or tolerances
- pass / fail decision

