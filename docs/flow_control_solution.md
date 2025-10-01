# Hooks 流控制解决方案

## 问题描述

原始问题：在用户交互期间，代理继续执行并输出到控制台，导致：
1. 用户输入提示被其他输出打断
2. 控制台显示顺序混乱
3. 用户体验不佳

## 解决方案架构

### 1. 执行状态管理 (`execution_state.py`)

- **ExecutionState 枚举**：定义执行状态（运行中、等待用户输入、暂停、完成）
- **ExecutionFlowController 类**：控制执行流程
  - 状态管理（设置/获取状态）
  - 暂停/恢复机制
  - 用户输入请求处理

### 2. Hooks 系统改造 (`generation_hooks.py`)

- **流控制集成**：所有 hooks 方法检查执行状态
- **暂停上下文管理器**：用户交互期间暂停执行
- **输出缓冲区管理**：确保输出清理和同步

### 3. 代理包装器 (`flow_controlled_agent.py`)

- **FlowControlledAgent 类**：包装代理以支持流控制
- **执行前检查**：运行前检查是否需要暂停

## 核心机制

### 暂停机制
```python
async with execution_controller.pause_execution():
    # 用户交互代码
    choice = await execution_controller.request_user_input(get_user_input)
```

### 等待恢复机制
```python
# 在所有 hooks 方法中
await execution_controller.wait_for_resume()
```

### 状态转换流程
1. RUNNING → PAUSED：用户交互开始
2. PAUSED → RUNNING：用户交互结束
3. RUNNING → WAITING_USER_INPUT：等待用户输入
4. WAITING_USER_INPUT → RUNNING：收到用户输入

## 使用方法

### 1. 在现有代码中集成
```python
from datus.cli.flow_controlled_agent import FlowControlledAgent

# 包装代理
agent = FlowControlledAgent.wrap_agent(your_agent)
```

### 2. 在 hooks 中使用
```python
from datus.cli.execution_state import execution_controller

async def your_hook_method(self, context, agent, *args):
    # 等待执行恢复
    await execution_controller.wait_for_resume()

    # 你的逻辑
```

### 3. 用户交互场景
```python
async with execution_controller.pause_execution():
    # 清理输出
    await self._clear_output_and_show_prompt()

    # 获取用户输入
    choice = await execution_controller.request_user_input(input_func)
```

## 优势

1. **完全的流控制**：暂停期间不会有任何代理执行
2. **清晰的输出**：用户交互期间输出整洁
3. **状态管理**：明确的执行状态转换
4. **可扩展性**：可以扩展支持更复杂的交互模式
5. **向后兼容**：不破坏现有代码结构

## 注意事项

1. **性能影响**：暂停机制会增加轻微延迟
2. **调试考虑**：暂停期间可能影响调试输出
3. **异常处理**：确保暂停状态在异常情况下正确恢复
4. **并发安全**：使用 asyncio.Lock 确保状态安全

## 进一步改进

1. **超时机制**：为用户输入添加超时
2. **取消机制**：允许取消长时间运行的执行
3. **进度指示**：显示执行状态
4. **日志增强**：更详细的状态转换日志