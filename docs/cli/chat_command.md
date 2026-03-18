# Chat Command `/`

## 1. Overview

The Chat Command `/` is the heart of Datus-CLI. It enables you to converse with the AI agent in a multi-turn session, describe tasks in natural language, and receive reasoning steps and SQL code suggestions. Think of it as your copilot for exploring data, drafting SQL, and planning workflows — all directly in the CLI.

You can chat with Datus in any format — plain English, bullet points, or sketches of logic — and freely edit or follow up on its responses. The agent keeps track of your instructions and previous outputs, so you can iteratively refine results without starting over.

---

## 2. Basic Usage

Start a new chat session by entering `/` followed by your message:

```text
/ How many orders were placed last week?
```

The agent will respond with its reasoning process and a proposed SQL query. You can then follow up naturally:

```text
/ Filter only for VIP customers
```

Datus streams the output as it thinks — showing each action's execution result in real time. If the result contains SQL, it will:

- Automatically highlight the SQL in the output
- Copy the SQL to your clipboard for quick use
- Finally produce a Markdown-formatted summary of the result

![Reasoning progress](../assets/reasoning_progress.png)

![Result of query](../assets/result_query.png)

![Details of function calling](../assets/function_calling_details.png)

---

## 3. Advanced Features

### Context Injection

Context Injection allows you to pull existing tables, metrics or files into your conversation. There are two ways to do this:

#### Browse Mode
Type `@` and press Tab to browse your context tree step by step. You can navigate by category (table / file / metrics) and drill down the directory-like structure to select the exact item you need.

#### Fuzzy Search Mode
Type `@` followed by some keywords, then press Tab to trigger a fuzzy search. Datus will suggest context items ranked by textual similarity, letting you quickly find what you need without knowing the exact path.

This is the fastest way to ground your prompts with precise context.

![Context injection browse mode](../assets/context_browse.png)

![Context injection fuzzy search](../assets/context_fuzzy.png)

### Interrupt Execution

Press **ESC** or **Ctrl+C** while the agent is running to gracefully interrupt the current execution. The agent will finish its current step, then stop and return control to you.

After an interrupt, the session remains intact — you can continue typing new instructions, refine your question, or provide additional context. Nothing is lost.

### Toggle Trace Display

Press **Ctrl+O** while the agent is running to toggle the trace display mode between **compact** (progress only) and **verbose** (full step details). This lets you control how much detail you see during execution without interrupting the agent.

### Session Commands

- `.clear`: Clear the current session context and start fresh
- `.compact`: Compress previous turns to reduce memory usage while preserving context
  - Auto-trigger: `.compact` will run automatically when the model context usage exceeds 90%, so you can continue chatting without hitting limits
- `.chat_info`: Show the current active context (messages, tables, metrics)
- `.resume [session_id]`: Resume a previous chat session
  - Without arguments: displays a table of all available sessions (sorted by last modified time), then prompts you to pick one by number
  - With a session ID: directly resumes that specific session
  - After resuming, the full conversation history is replayed so you can see what was discussed, and you can continue chatting from where you left off
  - If the session has high token usage (>50k tokens), a hint is shown suggesting `.compact` to reduce context size
- `.rewind [turn_number]`: Rewind the current session to a specific user turn, creating a new branched session
  - Displays a numbered table of all user turns in the current session
  - You select a turn number — the session is cloned up to (and including) that turn's assistant response, producing a new session
  - The original session is preserved; the rewound copy becomes your active session
  - Without arguments: shows the turn table and prompts for a turn number
  - With a turn number: directly rewinds to that turn
  - Useful when you want to retry a question with different phrasing or explore an alternative path without losing the original conversation