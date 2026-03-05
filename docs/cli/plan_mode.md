# Plan Mode User Guide(Beta)

## Overview

Plan Mode is an interactive workflow feature that helps you break down complex tasks into manageable steps. Instead of executing commands immediately, the AI first creates a detailed execution plan that you can review, modify, and control.

## How It Works

Plan Mode follows a three-phase workflow:

1. **Plan Generation** - AI analyzes your request and creates a step-by-step plan
2. **User Confirmation** - You review and choose how to execute the plan
3. **Execution** - Steps are executed based on your chosen mode

## Getting Started

To use Plan Mode, press **Shift+Tab** to toggle plan mode on/off before submitting your message. When enabled, the system will automatically enter planning workflow instead of executing immediately.

## Workflow Phases

### Phase 1: Plan Generation

When you submit a request in plan mode, the AI will:

- Analyze your requirements
- Break down the task into clear, actionable steps
- Display the proposed execution plan

Example output:
```text
Plan Generated Successfully!
Execution Plan:
  1. Query the customer database for recent orders
  2. Calculate total revenue by product category
  3. Generate summary report with visualizations
  4. Export results to CSV file
```

### Phase 2: Choose Execution Mode

After the plan is generated, you'll be prompted to select an execution mode:

```text
CHOOSE EXECUTION MODE:

  1. Manual Confirm - Confirm each step
  2. Auto Execute - Run all steps automatically
  3. Revise - Provide feedback and regenerate plan
  4. Cancel
```

#### Option 1: Manual Confirm Mode

- **Best for**: Tasks requiring careful review at each step
- **How it works**: Before each step execution, you'll see:
  - Full plan with progress indicators (✓ completed, ▶ current, ○ pending)
  - The next step to be executed
  - Options to continue, switch to auto mode, revise, or cancel

Example interaction:
```text
Plan Progress:
  ✓ 1. Query the customer database for recent orders
  ▶ 2. Calculate total revenue by product category
  ○ 3. Generate summary report with visualizations
  ○ 4. Export results to CSV file

Next step: Calculate total revenue by product category
Options:
  1. Execute this step
  2. Execute this step and continue automatically
  3. Revise remaining plan
  4. Cancel

Your choice (1-4) [1]:
```

#### Option 2: Auto Execute Mode

- **Best for**: Well-defined tasks that don't require step-by-step review
- **How it works**: All steps execute automatically with minimal interruption
- **Note**: You can still cancel execution at any point

Example interaction:
```text
Plan Progress:
  ✓ 1. Query the customer database for recent orders
  ▶ 2. Calculate total revenue by product category
  ○ 3. Generate summary report with visualizations
  ○ 4. Export results to CSV file

Auto Mode: Calculate total revenue by product category
Execute? (y/n) [y]:
```

#### Option 3: Revise Plan

- **Best for**: When the initial plan doesn't meet your needs
- **How it works**:
  - Provide feedback on what should be changed
  - AI regenerates the plan incorporating your feedback
  - Already completed steps are preserved
  - You can revise multiple times until satisfied

Example:
```text
Feedback for replanning: Add data validation step before calculation

Replanning with feedback: Add data validation step before calculation
[New plan generated with validation step included]
```

#### Option 4: Cancel

- Stops the entire workflow
- No steps will be executed
- Safe exit from plan mode

## Progress Tracking

During execution, the system shows real-time progress with visual indicators:

- **✓** (green checkmark) - Completed steps
- **▶** (yellow arrow) - Current step being executed
- **○** (white circle) - Pending steps

## Mid-Execution Options

While executing in manual mode, you can:

1. **Continue step-by-step** - Maintain control over each action
2. **Switch to auto mode** - Speed up remaining steps
3. **Revise remaining plan** - Adjust upcoming steps based on current results
4. **Cancel execution** - Stop the workflow

## Best Practices

### When to Use Manual Mode

- Complex database operations
- Tasks with potential side effects
- Learning new workflows
- Critical operations requiring validation

### When to Use Auto Mode

- Routine, well-tested operations
- Report generation
- Data exports
- Multi-step queries you're confident about


## Error Handling

If a step fails during execution:

- The system marks it as failed
- Execution pauses in manual mode
- You can revise the plan to address the error
- Previously completed steps are preserved


## Summary

Plan Mode gives you control over complex workflows by:

- Breaking tasks into clear steps
- Letting you review before execution
- Providing flexibility during execution
- Preserving progress when revising plans

Choose manual mode when you need control, auto mode when you're confident, and don't hesitate to revise when the plan needs adjustment.
