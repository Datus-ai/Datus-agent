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

### Session Commands

- `.clear`: Clear the current session context and start fresh
- `.compact`: Compress previous turns to reduce memory usage while preserving context
  - Auto-trigger: `.compact` will run automatically when the model context usage exceeds 90%, so you can continue chatting without hitting limits
- `.chat_info`: Show the current active context (messages, tables, metrics)