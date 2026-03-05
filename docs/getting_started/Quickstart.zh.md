# 快速开始

几分钟内即可上手 Datus Agent。本指南将带你完成安装、配置和首次体验。

## 步骤 1：安装与配置

### 安装 Python 3.12

Datus 需要 Python 3.12 运行环境，可按喜好选择以下方式：

=== "Conda"

    ```bash
    conda create -n datus python=3.12
    conda activate datus
    ```

=== "virtualenv"

    ```bash
    virtualenv datus --python=python3.12
    source datus/bin/activate
    ```

=== "uv"

    ```bash
    uv venv --python 3.12
    source .venv/bin/activate
    ```

### 安装 Datus Agent

!!! note
    请确保您的 pip 版本与 Python 3.12 兼容。

    如需升级 pip,可以使用以下命令:
    ```bash
    python -m ensurepip --upgrade
    python -m pip install --upgrade pip setuptools wheel
    ```

=== "稳定版"

    ```bash
    pip install datus-agent
    ```

=== "测试版"

    ```bash
    pip install --no-deps -i https://test.pypi.org/simple/ datus-agent
    ```

### 初始化配置

运行初始化命令：

```bash
datus-agent init
```

初始化流程将引导你完成：

**1. 大模型配置** —— 选择并配置首选的大模型服务（OpenAI、DeepSeek、Claude、Kimi、Qwen 等）

**2. 命名空间设置** —— 连接到你的数据库。想快速体验，可使用演示数据库：

!!! tip "演示数据库"
    Datus 提供预配置的 DuckDB 演示库，便于测试。

    **连接字符串：** `~/.datus/sample/duckdb-demo.duckdb`

**3. 工作区配置** —— 指定 SQL 文件目录（默认：`~/.datus/workspace`）

**4. 知识库（可选）** —— 初始化向量数据库，用于存储元数据和Reference SQL

> **注意**
> 如果在此步骤卡住或失败，可能是国内访问 HuggingFace 官方源的问题，建议配置国内镜像地址 `export HF_ENDPOINT=https://hf-mirror.com`。

完成设置后，你就可以启动 Datus 了！

## 步骤 2：启动 Datus CLI

使用已配置的命名空间启动 Datus CLI：

!!! tip "配置提示"
    你可以在 `agent.yml` 中添加多个命名空间，以连接不同的数据库。详见我们的[配置指南](../configuration/introduction.md)。

```bash title="Terminal"
datus-cli --namespace duckdb-demo
```
```{ .yaml .no-copy }
Initializing AI capabilities in background...

Datus - AI-powered SQL command-line interface
Type '.help' for a list of commands or '.exit' to quit.

Namespace duckdb selected
Connected to duckdb using database duckdb
Context: Current: database: duckdb
Type SQL statements or use ! @ . commands to interact.
Datus>
```

## 步骤 3：开始使用 Datus

!!! tip
    你可以像在普通 SQL 编辑器中那样执行 SQL。

列出所有表：

```bash title="Terminal"
Datus> .tables
```
```{ .yaml .no-copy }
Tables in Database duckdb-demo
+---------------------+
| Table Name          |
+=====================+
| bank_failures       |
| boxplot             |
| calendar            |
| candle              |
| christmas_cost      |
| companies           |
| country_stats_scatter|
| gold_vs_bitcoin     |
| japan_births_deaths |
| japan_population    |
| metrics             |
| niger_population    |
| quotes              |
| radar               |
| sankey              |
| search_trends       |
| tree                |
+---------------------+
```

你可以提出任何问题。以 `gold_vs_bitcoin` 表为例，先查看其结构：

```bash title="Terminal"
Datus> desc gold_vs_bitcoin
```
```{ .yaml .no-copy }
+------------------+------------------+------------------+------------------+------------------+------------------+
| column_name      | column_type      | null             | key              | default          | extra            |
+==================+==================+==================+==================+==================+==================+
| time             | TIMESTAMP        | YES              | None             | None             | None             |
| gold             | DOUBLE           | YES              | None             | None             | None             |
| bitcoin          | DOUBLE           | YES              | None             | None             | None             |
+------------------+------------------+------------------+------------------+------------------+------------------+
Returned 3 rows in 0.04 seconds
```

!!! tip
    若想与智能体对话，在命令前加 `/` 并使用自然语言即可。

想要了解黄金与比特币之间的相关性，可以直接向 Datus 提问：

```bash title="Terminal"
Datus> /Detailed analysis of gold–Bitcoin correlation.
```

Datus 会展示其思考过程，说明分析步骤，并给出详细报告。

??? example "详细分析输出"
    ```{ .yaml .no-copy }
    Creating new chat session...
    Processing chat request...
    +---------------------------------------------------------------------------------------------------------------------------------------+
    |                                                    Action Stream                                                                      |
    +=======================================================================================================================================+
    | 🔧 Tool call - search_files('{"path": "/Users/yt/.datus/workspace", "pattern": ...') - ✓                                            |
    |     No matches found (0.5s)                                                                                                          |
    | 💬 Thinking: Let me check the directory structure to see what's available:                                                           |
    | 🔧 Tool call - list_directory('{"path": "/Users/yt/.datus/workspace"}') - ✓ (0.5s)                                                   |
    | 💬 Thinking: Now let me analyze the gold-Bitcoin correlation by examining the data and performing a comprehensive correlation        |
    | analysis:                                                                                                                            |
    | 🔧 Tool call - read_query('{"sql": "SELECT \n    COUNT(*) as total_records,\n...') - ✓                                               |
    |     ✓ 1 rows (0.5s)                                                                                                                  |
    | 💬 Thinking: Now let me calculate the correlation coefficient and perform a detailed analysis:                                       |
    | 🔧 Tool call - read_query('{"sql": "SELECT \n    -- Correlation coefficient\n...') - ✓                                               |
    |     ✓ 1 rows (0.5s)                                                                                                                  |
    | 💬 Thinking: Let me get more detailed statistics and time-based analysis:                                                            |
    | 🔧 Tool call - read_query('{"sql": "WITH daily_aggregates AS (\n    SELECT \n...') - ✓                                               |
    |     ✓ 1 rows (0.5s)                                                                                                                  |
    | 💬 Thinking: Let me check the distribution of the data and look for any patterns:                                                    |
    | 🔧 Tool call - read_query('{"sql": "SELECT \n    -- Price movement analysis\n...') - ✓                                               |
    |     ✗ Failed:(error_code=500006, e...) (0.5s)                                                                                        |
    | 💬 Thinking: Let me fix the query to properly analyze price movement patterns:                                                       |
    | 🔧 Tool call - read_query('{"sql": "WITH price_changes AS (\n    SELECT \n   ...') - ✓                                               |
    |     ✓ 5 rows (0.5s)                                                                                                                  |
    | 💬 Thinking: Now let me analyze the strength of the correlation across different time periods and price levels:                      |
    | 🔧 Tool call - read_query('{"sql": "WITH price_quartiles AS (\n    SELECT \n ...') - ✓                                               |
    |     ✓ 16 rows (0.5s)                                                                                                                 |
    | 💬 Thinking: Let me also analyze the volatility and risk characteristics:                                                            |
    | 🔧 Tool call - read_query('{"sql": "SELECT \n    -- Risk and return metrics\n...') - ✓                                               |
    |     ✗ Failed:(error_code=500006, e...) (0.5s)                                                                                        |
    | 💬 Thinking: Let me correct the query with the proper function names:                                                                |
    | 🔧 Tool call - read_query('{"sql": "SELECT \n    -- Risk and return metrics\n...') - ✓                                               |
    |     ✓ 1 rows (0.5s)                                                                                                                  |
    | 💬 Thinking: Finally, let me examine the time series data to understand the temporal patterns:                                       |
    | 🔧 Tool call - read_query('{"sql": "SELECT \n    DATE(time) as trading_date,\...') - ✓                                               |
    |     ✓ 1 rows (0.5s)                                                                                                                  |
    | 💬 Chat interaction completed successfully                                                                                           |
    +---------------------------------------------------------------------------------------------------------------------------------------+
    ```

**生成的 SQL：**

```sql title="Generated Query"
-- Generated SQL (copied)
SELECT
    -- Overall correlation and basic stats
    CORR(gold, bitcoin) as correlation_coefficient,
    COUNT(*) as total_observations,
    AVG(gold) as avg_gold_price,
    AVG(bitcoin) as avg_bitcoin_price,
    STDDEV(gold) as gold_volatility,
    STDDEV(bitcoin) as bitcoin_volatility,
    COVAR_POP(gold, bitcoin) as covariance,

    -- Price movement patterns
    (SELECT COUNT(*) FROM (
        SELECT
            CASE
                WHEN gold_change > 0 AND bitcoin_change > 0 THEN 'Both Up'
                WHEN gold_change < 0 AND bitcoin_change < 0 THEN 'Both Down'
                WHEN gold_change > 0 AND bitcoin_change < 0 THEN 'Gold Up, Bitcoin Down'
                WHEN gold_change < 0 AND bitcoin_change > 0 THEN 'Gold Down, Bitcoin Up'
            END as pattern
        FROM (
            SELECT
                gold - LAG(gold) OVER (ORDER BY time) as gold_change,
                bitcoin - LAG(bitcoin) OVER (ORDER BY time) as bitcoin_change
            FROM gold_vs_bitcoin
        ) WHERE gold_change IS NOT NULL
    ) WHERE pattern = 'Both Up') as both_up_count,

    (SELECT COUNT(*) FROM (
        SELECT
            CASE
                WHEN gold_change > 0 AND bitcoin_change > 0 THEN 'Both Up'
                WHEN gold_change < 0 AND bitcoin_change < 0 THEN 'Both Down'
                WHEN gold_change > 0 AND bitcoin_change < 0 THEN 'Gold Up, Bitcoin Down'
                WHEN gold_change < 0 AND bitcoin_change > 0 THEN 'Gold Down, Bitcoin Up'
            END as pattern
        FROM (
            SELECT
                gold - LAG(gold) OVER (ORDER BY time) as gold_change,
                bitcoin - LAG(bitcoin) OVER (ORDER BY time) as bitcoin_change
            FROM gold_vs_bitcoin
        ) WHERE gold_change IS NOT NULL
    ) WHERE pattern = 'Both Down') as both_down_count
FROM gold_vs_bitcoin
```

**分析结果：**

??? success "黄金-比特币相关性分析"
    ```{ .yaml .no-copy }
    +===============================================================================================================+
    |                                        Gold-Bitcoin Correlation Analysis                                     |
    +===============================================================================================================+

                                                          📊 Overall Correlation

    • Correlation Coefficient: -0.225 (Weak negative correlation)
    • Total Observations: 954 data points
    • Time Period: Single day - December 1, 2024 (16:00 to 16:15)


                                                            💰 Price Statistics

    • Average Gold Price: 0.684
    • Average Bitcoin Price: 0.785
    • Gold Volatility: 0.200
    • Bitcoin Volatility: 0.233
    • Covariance: -0.0105


                                                        📈 Price Movement Patterns

    Based on 953 observed price changes:

    • Both Assets Moving Up: 216 times (22.67%)
    • Both Assets Moving Down: 203 times (21.30%)
    • Gold Up, Bitcoin Down: 189 times (19.83%)
    • Gold Down, Bitcoin Up: 190 times (19.94%)


                                                              🎯 Key Insights

                                                       1. Weak Negative Correlation

    The -0.225 correlation suggests these assets tend to move in opposite directions more often than together, though the relationship
    is weak.

                                                       2. Similar Movement Patterns

    Both assets show remarkably similar movement patterns:

    • Approximately 44% of the time they move in the same direction
    • Approximately 40% of the time they move in opposite directions

                                                         3. Volatility Comparison

    Bitcoin shows slightly higher volatility (0.233) compared to Gold (0.200), which aligns with Bitcoin's reputation as a more
    volatile asset.

                                                       4. Risk-Adjusted Performance

    • Gold Sharpe Ratio: 3.42
    • Bitcoin Sharpe Ratio: 3.37 Both assets show strong risk-adjusted returns during this period.


                                                               ⚠️ Limitations

    This analysis is based on a very short time window (15 minutes of data). For more robust correlation analysis, data spanning
    multiple days/weeks would be needed to observe:

    • Longer-term correlation trends
    • Market regime changes
    • Impact of macroeconomic events
    • Seasonal patterns


                                                        💡 Investment Implications

    The weak negative correlation suggests these assets could provide some diversification benefits, though the relationship is not
    strong enough for reliable hedging strategies based on this limited dataset.
    ```

如果想查看每一步的详细信息，可在结束时输入 `y`：

```bash title="Terminal"
Would you like to check the details? (y/n): y
```

Datus 会在新窗口展示所有步骤详情。

## 步骤 4：进阶功能

### 上下文管理

!!! note
    前提是你已经初始化了用于元数据的知识库。

如果希望把文件、指标或数据表加入上下文，可使用 `@` 命令。手动检索不仅能为大模型提供更精确的上下文，还能缩小搜索范围，提高回答准确度。

```bash title="Terminal"
Datus> /@Table duckdb-demo.main.bank_failures
```

Datus 会自动分析该表，并将元数据加入上下文。

??? example "上下文分析输出"
    ```{ .yaml .no-copy }
    Processing chat request...
    +----------------------------------------------------------------------------------------------------------------------------------+
    |                                                    Action Stream                                                                 |
    +==================================================================================================================================+
    | 🟡 User: @Table duckdb-demo.main.bank_failures                                                                                  |
    | 💬 Generating response with tools...                                                                                            |
    | 💬 Thinking: I'll provide a comprehensive analysis of the bank_failures table. Let me start by examining the data structure     |
    | and then perform various analyses.                                                                                              |
    | 🔧 Tool call - describe_table('{"table_name": "bank_failures"}') - ✓                                                            |
    |     ✓ 7 columns (0.5s)                                                                                                          |
    | 💬 Thinking: Now let me perform a comprehensive analysis of the bank failures data:                                             |
    | 🔧 Tool call - read_query('{"sql": "-- Comprehensive Bank Failures Analysis\n...') - ✓                                          |
    |     ✗ Failed:(error_code=500006, e...) (0.5s)                                                                                   |
    | 💬 Thinking: Let me break this down into separate queries for a comprehensive analysis:                                         |
    | 🔧 Tool call - read_query('{"sql": "-- 1. Basic overview statistics\nSELECT \...') - ✓                                          |
    |     ✓ 1 rows (0.5s)                                                                                                             |
    | 🔧 Tool call - read_query('{"sql": "-- 2. Yearly analysis of bank failures\nS...') - ✓                                          |
    |     ✓ 14 rows (0.5s)                                                                                                            |
    | 🔧 Tool call - read_query('{"sql": "-- 3. State-wise analysis (top 15 states)...') - ✓                                          |
    |     ✓ 15 rows (0.5s)                                                                                                            |
    | 🔧 Tool call - read_query('{"sql": "-- 4. Largest bank failures (top 20 by as...') - ✓                                          |
    |     ✓ 20 rows (0.5s)                                                                                                            |
    | 🔧 Tool call - read_query('{"sql": "-- 5. Acquisition analysis (top acquirers...') - ✓                                          |
    |     ✓ 15 rows (0.5s)                                                                                                            |
    | 🔧 Tool call - read_query('{"sql": "-- 6. Asset size distribution analysis\nW...') - ✓                                          |
    |     ✓ 6 rows (0.5s)                                                                                                             |
    | 🔧 Tool call - read_query('{"sql": "-- 7. Recent failures (2023-2024)\nSELECT...') - ✓                                          |
    |     ✓ 8 rows (0.5s)                                                                                                             |
    | 💬 Chat interaction completed successfully                                                                                      |
    +----------------------------------------------------------------------------------------------------------------------------------+

    +==================================================================================================================================+
    |                                       🏦 Comprehensive Bank Failures Analysis (2008-2024)                                       |
    +==================================================================================================================================+

                                                           📊 Executive Summary

    • Total Failures: 545 banks failed between 2008-2024
    • Total Assets: $1.696 trillion in failed bank assets
    • Time Span: 16 years (January 2008 - October 2024)
    • Geographic Reach: 42 states, 419 cities affected
    • Average Bank Size: $3.11 billion


                                                        📈 Key Historical Patterns

                                                           Major Crisis Periods

    1 2008-2009 Financial Crisis: 166 failures, $938.1B assets
    • Peak: 2009 (140 failures, $169.5B assets)
    • Includes Bear Stearns ($395B) and Washington Mutual ($307B)
    2 2010-2014 Post-Crisis Cleanup: 270 failures, $150.0B assets
    • Peak: 2010 (157 failures - highest count)
    • Smaller institutions, regulatory cleanup
    3 2023 Banking Stress: 6 failures, $572.7B assets
    • Major failures: First Republic ($229B), Silicon Valley Bank ($209B), Signature Bank ($118B)


                                                        🗺️ Geographic Distribution

                                                      Top 5 States by Failure Count

    1 Georgia: 91 failures (16.7%) - $34.4B assets
    2 Florida: 73 failures (13.4%) - $39.2B assets
    3 Illinois: 66 failures (12.1%) - $37.7B assets
    4 California: 43 failures (7.9%) - $559.8B assets
    5 Minnesota: 23 failures (4.2%) - $3.2B assets

                                                       Top 5 States by Assets Lost

    1 California: $559.8B (33.0% of total)
    2 New York: $513.4B (30.3% of total)*
    3 Washington: $318.9B (18.8% of total)
    4 Illinois: $37.7B (2.2% of total)
    5 Florida: $39.2B (2.3% of total)

    *Includes Bear Stearns and Signature Bank


                                                          💰 Asset Size Analysis

                                                            Size Distribution


    Asset Range   Failures   % of Total   Total Assets   % of Assets   Avg Size
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    Under $100M   136        24.9%        $7.7B          0.5%          $56M
    $100M-$500M   275        50.5%        $66.5B         3.9%          $242M
    $500M-$1B     57         10.5%        $40.3B         2.4%          $707M
    $1B-$5B       55         10.1%        $119.6B        7.1%          $2.2B
    $5B-$10B      7          1.3%         $43.8B         2.6%          $6.3B
    Over $10B     15         2.8%         $1,418.2B      83.6%         $94.5B


    Key Insight: While 75% of failures were under $1B in assets, the 15 largest failures (>$10B) represent 83.6% of all failed assets.


                                                         🏆 Largest Bank Failures

    1 Bear Stearns (2008) - $395.0B (J.P. Morgan Chase)
    2 Washington Mutual (2008) - $307.0B (JPMorgan Chase)
    3 First Republic Bank (2023) - $229.0B (JPMorgan Chase)
    4 Silicon Valley Bank (2023) - $209.0B (First Citizens)
    5 Signature Bank (2023) - $118.4B (Flagstar Bank)


                                                         🏛️ Acquisition Landscape

                                                        Top Acquiring Institutions

    1 U.S. Bank, N.A.: 13 acquisitions ($38.7B)
    2 State Bank and Trust: 10 acquisitions ($3.5B)
    3 Ameris Bank: 9 acquisitions ($1.9B)
    4 Stearns Bank, N.A.: 9 acquisitions ($1.7B)
    5 First-Citizens Bank: 7 acquisitions ($7.8B)

                                                             Notable Patterns

    • JPMorgan Chase: Acquired 3 institutions but captured $930B+ in assets
    • 20 banks: Had no acquirer (insured depositors paid directly)
    • Consolidation: Larger banks acquired most failed institutions


                                                       📅 Recent Trends (2023-2024)

                                                           2023 Major Failures

    • First Republic Bank: $229B (May 2023)
    • Silicon Valley Bank: $209B (March 2023)
    • Signature Bank: $118B (March 2023)
    • Silvergate Bank: $16B (March 2023)

                                                              2024 Failures

    • Republic First Bancorp: $6B (April 2024)
    • First National Bank of Lindsay: $108M (October 2024)


                                                             🎯 Key Insights

                                                      1. Systemic Risk Concentration

    • 15 largest failures (2.8% of total) represent 83.6% of failed assets
    • "Too big to fail" remains a significant concern

                                                       2. Geographic Vulnerability

    • Southeast (GA, FL) had highest failure counts
    • West Coast (CA, WA) had largest asset concentrations

                                                          3. Temporal Clustering

    • Clear crisis periods: 2008-2009, 2010-2014, 2023
    • Regulatory changes followed each crisis period

                                                             4. Size Matters

    • Most failures were small banks
    • But systemic risk comes from very large institutions

                                                         5. Acquisition Patterns

    • Large banks benefit from crisis acquisitions
    • Significant industry consolidation


                                                      ⚠️ Risk Management Implications

    1 Regulatory Focus: Largest institutions require heightened supervision
    2 Regional Monitoring: Southeast banking markets need closer oversight
    3 Crisis Preparedness: Friday intervention patterns suggest need for rapid response capabilities
    4 Systemic Monitoring: Asset concentration remains critical risk indicator

    This analysis provides a comprehensive view of US bank failures over 16 years, highlighting patterns, risks, and regulatory
    implications for the banking sector.
    Would you like to check the details? (y/n): n
    ```

!!! tip
    需要更多命令参考与用法，请查看 [CLI](../cli/introduction.md)，或在终端输入 `.help`。

## 下一步

在完成基础体验后，可以继续探索以下功能：

- **[上下文数据工程](./contextual_data_engineering.md)** —— 学习如何将数据资产用作上下文
- **[配置指南](../configuration/introduction.md)** —— 连接自有数据库并自定义设置
- **[CLI 参考手册](../cli/introduction.md)** —— 掌握全部命令与选项
- **[语义层适配器](../adapters/semantic_adapters.md)** —— 使用 datus-semantic-metricflow 构建与查询指标



