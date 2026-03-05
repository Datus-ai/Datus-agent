# 计划模式用户指南（Beta）

## 概览

计划模式是一种交互式工作流特性，帮助你将复杂任务拆解为可管理的步骤。AI 会先生成详细执行计划，供你审阅、修改并控制执行节奏，而不是立即执行命令。

## 工作方式

计划模式包含三个阶段：

1. **计划生成** —— AI 分析请求并生成逐步计划
2. **用户确认** —— 你审阅计划并选择执行方式
3. **执行阶段** —— 根据所选模式逐步执行计划

## 快速上手

提交消息前，按 **Shift+Tab** 切换计划模式开/关。启用后，系统会进入规划流程，而不是立刻执行。

## 工作流阶段

### 阶段一：计划生成

在计划模式下提交请求后，AI 将：

- 分析你的需求
- 将任务拆解为清晰可执行的步骤
- 展示建议的执行计划

示例输出：
```text
Plan Generated Successfully!
Execution Plan:
  1. Query the customer database for recent orders
  2. Calculate total revenue by product category
  3. Generate summary report with visualizations
  4. Export results to CSV file
```

### 阶段二：选择执行模式

生成计划后，系统会提示你选择执行模式：
```text
CHOOSE EXECUTION MODE:

  1. Manual Confirm - Confirm each step
  2. Auto Execute - Run all steps automatically
  3. Revise - Provide feedback and regenerate plan
  4. Cancel
```

#### 选项 1：手动确认模式

- **适用场景**：需要逐步审查的任务
- **执行方式**：在每个步骤执行前，你会看到：
  - 带进度标记的完整计划（✓ 完成、▶ 当前、○ 待执行）
  - 即将执行的步骤说明
  - 继续、切换自动模式、重新规划或取消的操作选项

交互示例：
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

#### 选项 2：自动执行模式

- **适用场景**：定义清晰、无需逐步审阅的任务
- **执行方式**：所有步骤自动串行执行，最少打断
- **提示**：随时可以中断执行

交互示例：
```text
Plan Progress:
  ✓ 1. Query the customer database for recent orders
  ▶ 2. Calculate total revenue by product category
  ○ 3. Generate summary report with visualizations
  ○ 4. Export results to CSV file

Auto Mode: Calculate total revenue by product category
Execute? (y/n) [y]:
```

#### 选项 3：重新规划

- **适用场景**：初始计划不符合预期
- **执行方式**：
  - 提供需要调整的反馈
  - AI 根据反馈重新生成计划
  - 已完成的步骤会保留
  - 可多次迭代直至满意

示例：
```text
Feedback for replanning: Add data validation step before calculation

Replanning with feedback: Add data validation step before calculation
[New plan generated with validation step included]
```

#### 选项 4：取消

- 停止整个工作流
- 不执行任何步骤
- 安全退出计划模式

## 进度追踪

执行过程中，系统会用可视化符号实时展示进度：

- **✓**（绿色对勾）—— 已完成步骤
- **▶**（黄色箭头）—— 正在执行的步骤
- **○**（白色圆点）—— 待执行步骤

## 执行中的选项

手动模式下执行时，可以：

1. **逐步继续** —— 保持对每个步骤的掌控
2. **切换到自动模式** —— 加速后续步骤
3. **重新规划剩余步骤** —— 根据当前结果调整后续动作
4. **取消执行** —— 终止工作流

## 最佳实践

### 何时选择手动模式

- 复杂的数据库操作
- 可能产生副作用的任务
- 学习新工作流时
- 需要验证的关键操作

### 何时选择自动模式

- 日常、已验证的任务
- 报表生成
- 数据导出
- 自信掌握的多步骤查询

## 错误处理

若执行过程中某个步骤失败：

- 系统会标记该步骤失败
- 手动模式下执行会暂停
- 你可以重新规划以解决错误
- 已完成的步骤会被保留

## 总结

计划模式通过以下方式帮助你掌控复杂工作流：

- 将任务拆解为清晰步骤
- 在执行前提供审阅机会
- 执行过程中保持灵活度
- 在重新规划时保留已完成进度

当你需要精细掌控时选择手动模式，确信结果可靠时使用自动模式；若计划不理想，随时重生成更合适的版本。

