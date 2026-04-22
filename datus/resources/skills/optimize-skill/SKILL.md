---
name: optimize-skill
description: Optimize and improve existing Datus skills. Use when users want to edit a skill, improve its instructions, optimize its description for better triggering, or analyze skill performance based on usage sessions. Trigger phrases include "optimize skill", "improve skill", "edit skill", "fix skill", "skill not triggering".
tags: [skill, optimization, improvement]
version: "1.0.0"
user_invocable: false
allowed_agents:
  - gen_skill
---

# Optimize Skill

Guide for analyzing, editing, and optimizing existing Datus skills based on real usage data.

## Step 1: Identify Target Skill

Use `load_skill(skill_name="<name>")` to load the current SKILL.md content.

Present to the user:
- Current name, description, tags
- Summary of key instruction sections
- Whether it has scripts, references, or other resources

Ask the user what aspect needs improvement (or proceed to analysis if they said "optimize").

## Step 2: Find Usage Sessions

Search for sessions where this skill was invoked. Look at the action history and tool call records:

- Search for `load_skill` calls with this skill name in recent action histories
- Identify which agent nodes loaded this skill
- Find the corresponding `execute_stream` sessions

This gives real-world data on how the skill was actually used.

## Step 3: Analyze Tool Call Patterns

From the usage sessions, identify:

- **Which tools were called** after loading the skill — does the agent follow the skill's instructions?
- **What failed** — tool errors, retries, dead ends
- **Where the agent got stuck** — excessive tool calls, circular patterns, repeated queries
- **Repeated work patterns** — if every session writes similar helper scripts, the skill should bundle them
- **Unused instructions** — parts of the skill the agent consistently ignores
- **Missing guidance** — situations where the agent improvises because the skill doesn't cover them

## Step 4: Generate Optimization Suggestions

Based on analysis, propose specific changes:

- **Improve instructions**: Clarify ambiguous guidance that caused agent confusion
- **Add missing examples**: Where the agent had to guess, add concrete input/output pairs
- **Bundle repeated scripts**: If the agent keeps creating the same helper, put it in `scripts/`
- **Remove dead weight**: Instructions the agent ignores aren't pulling their weight
- **Fix gaps**: Add coverage for scenarios where the agent got stuck

## Step 5: Rewrite

Generate the improved SKILL.md:

1. Show a summary of proposed changes to the user
2. Use `ask_user` to confirm the changes
3. Write the updated file via `write_file` (keep the same path returned by `load_skill`; must stay under `.datus/skills/...` or `~/.datus/skills/...`)
4. Call `validate_skill` to verify
5. Show a before/after comparison of key changes

## Step 6: Description Optimization

The description field determines whether the agent invokes a skill. Optimize it:

- **Be assertive**: "Use whenever X" not "Can be used for X"
- **Include trigger contexts**: What user phrases should activate this skill
- **Adjacent keywords**: Related terms the user might use
- **Edge cases**: Phrases that SHOULD trigger vs phrases that should NOT
- **Mental test**: "Would the agent correctly decide to use this skill for [scenario]?"

Example transformation:
- Before: "Helps with SQL optimization"
- After: "Analyze and optimize SQL queries for performance. Use whenever the user mentions slow queries, query optimization, EXPLAIN plans, index suggestions, or database performance tuning, even if they don't explicitly ask for optimization."

## Improvement Principles

- **Generalize from feedback**: Skills are used many times across many prompts. Avoid overfitting to specific examples.
- **Keep the prompt lean**: Remove instructions that aren't pulling their weight. Read transcripts, not just outputs — if the skill makes the agent waste time on unproductive steps, remove those instructions.
- **Explain the why**: Instead of rigid MUSTs, explain reasoning. Today's LLMs are smart — when given good context they go beyond rote instructions.
- **Look for repeated work**: If the agent consistently writes similar helper scripts across sessions, bundle them in `scripts/`.
- **Theory of mind**: Try to understand the task from the user's perspective and transmit that understanding into the instructions.
