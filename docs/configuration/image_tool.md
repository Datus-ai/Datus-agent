# Image Reading

Ask the agent to inspect a local screenshot, chart, table, diagram, or photo:

```text
Read images/dashboard.png and explain what the chart shows.
```

The agent calls `filesystem_tools.read_image(path)` and sends the image to the currently selected model. There is no separate vision model or additional API key. Datus does not pre-check model capabilities. If the model rejects image input, the request follows the existing model error handling. The image tool result and conversation history are retained. Subsequent requests may fail again with the same model; you can switch to a model that accepts images or start a new session.

## Availability and permissions

`read_image` is included in the filesystem tool group and uses the same path permissions as `read_file`. Paths are relative to the project root or absolute. For an agent with an explicit tool list, include `filesystem_tools.read_image` or `filesystem_tools.*`.

The file must exist on the machine running the agent. This tool does not upload attachments, read clipboard images, or download image URLs.

## Supported files

| Property | Limit |
|---|---|
| Formats | Static PNG, JPEG, WebP; validated from file contents |
| Source file size | 10 MiB |
| Source dimensions | 25 million pixels |
| Image sent to the model | Orientation corrected; longest side at most 2048 pixels |
| Encoded image size | At most 3 MiB before base64 encoding; further resized if necessary |

Animated images, PDFs, SVGs, corrupt files, and files outside these limits return a tool error. The original file is unchanged. Small text may become harder to read after resizing; crop the relevant area into a separate image when needed.

## Conversation history

Images are retained in the local session database for follow-up questions and session restoration. Older images may be replaced by a source-path marker during context compaction; the agent can call `read_image` again while the original file remains available. The CLI, exported conversation text, logs, and traces contain image metadata or omission markers instead of image bytes.
