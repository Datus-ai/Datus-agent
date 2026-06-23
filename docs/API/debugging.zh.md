# API 调试

本页收录用于诊断 API 服务器异常的工具，目前只有一个。

## SIGUSR1 任务转储

### 适用场景

有时一个 API 请求迟迟不返回，但服务器其实还活着：`GET /health` 正常响应，进程 CPU 也没有被打满，可
`/chat/stream` 调用就是一直挂在那里。这几乎总是 **某个协程被挂起**，而不是事件循环被阻塞——某个 `await`
在等待一个永远不会到来的结果。常见原因是某个工具在等待交互确认（例如权限提示），而在 REST API 这种非交互
界面上没有人能回应它。

由于事件循环本身仍在运行，你可以让服务器遍历所有存活的 `asyncio` 任务，打印出每个任务当前挂起的位置。

### 触发转储

API 服务器在启动时会安装一个 `SIGUSR1` 处理器。向服务器进程发送该信号，会把所有存活异步任务的快照写入日志：

```bash
# 找到服务器 PID（daemon 模式会把它写入 PID 文件）
cat ~/.datus/run/datus-agent-api.pid

# 触发转储
kill -USR1 <pid>
```

转储以 `WARNING` 级别输出到服务器日志。daemon 模式下即 daemon 日志文件（默认 `logs/datus-agent-api.log`，
或 `--daemon-log-file` 指定的路径）；前台模式下则与服务器其他日志写到同一处。

`SIGUSR1` 不被 uvicorn 或 Datus 用作其他用途，因此可以反复发送——每次信号都会生成一份新的快照。

### 解读转储

每个存活任务会以两种视图呈现：

```
=== async task dump: 3 live task(s) ===

--- Task 'Task-42' done=False coro=chat_stream ---
  File ".../datus/api/service.py", line 312, in chat_stream
    async for event in agent.run(...):
  ...
await chain (innermost last):
  File ".../datus/tools/permission/broker.py", line 88, in request
    decision = await self._future
=== end async task dump ===
```

- **`Task.print_stack` 输出**——直到被直接挂起的协程为止的调用链。它停在第一个 `await`，不会跨越
  `async for`（异步生成器）边界继续往下走。
- **`await chain (innermost last)`**——真正有诊断价值的视图。它沿着
  `cr_await` / `ag_await` / `gi_yieldfrom` 链路跨越异步生成器和嵌套 await 边界，一直追到**最内层被挂起的
  栈帧**——也就是协程实际卡住的那一行（例如 `await broker.request(...)`）。

await chain 的最内层栈帧会告诉你请求究竟在等待什么。上面的例子中，请求挂在一个永远不会被解析的权限 broker
future 上，印证了「交互永远不会被回应」的判断。

如果某个任务在等待一个没有可下钻栈帧的裸 `Future`，await chain 会显示
`<no descendable frames; coroutine may be awaiting a bare Future>`——这依然有用，因为它排除了异步生成器卡死
的可能。

### 平台说明

- Windows 上不存在 `SIGUSR1`。处理器安装会在那里优雅跳过（返回 `False` 并以 `DEBUG` 记录日志），不影响
  服务器启动。在 Windows 上请改用前台日志或挂接调试器。
- 转储永远不会抛异常：任何内省错误都会被折叠进转储文本，因此触发它绝不会导致服务器崩溃。
