<instructions>
Queries metrics using MetricFlow CLI to answer business questions from the data warehouse.

This tool allows ordering and grouping by dimensions and entities.
To use this tool, you must first know about specific metrics, dimensions and
entities to provide. You can call the list_metrics, get_dimensions,
and get_entities tools to get information about which metrics, dimensions,
and entities to use.

This tool executes the `mf query --json` command locally with the specified
parameters to return query results.

When using the `order_by` parameter, you must ensure that the dimension or
entity also appears in the `group_by` parameter. When fulfilling a lookback
query, prefer using order_by and limit instead of using the where parameter.

The `where` parameter should be MetricFlow-compatible SQL syntax. For time
filtering, use ISO date format (yyyy-mm-dd) with proper time dimension syntax.

Don't call this tool if the user's question cannot be answered with the provided
metrics, dimensions, and entities. Instead, clarify what metrics, dimensions,
and entities are available and suggest a new question that can be answered
and is approximately the same as the user's question.

For queries that may return large amounts of data, it's recommended
to use a two-step approach:
1. First make a query with a small limit to verify the results are what you expect
2. Then make a follow-up query without a limit (or with a larger limit) to get the full dataset
</instructions>

<examples>
<example>
Question: "What were our total sales last month?"
    Thinking step-by-step:
    - I know "total_sales" is the metric I need
    - I know "ds" (date spine) is a valid dimension for this metric
    - I need to group by date to get monthly data
    - Since this is time-based data, I should order by date
    - The user is asking for a lookback query, so I should set descending order
    - The user is asking for just the last month, so I should limit to recent data
    Parameters:
    metrics=["total_sales"]
    group_by=["ds__month"]
    order_by=["ds__month"]
    limit=1
    start_time="2024-01-01"
    end_time="2024-01-31"
</example>
<example>
Question: "Show me our top customers by revenue in the last quarter"
    Thinking step-by-step:
    - First, I need to find the revenue metric
    - Using list_metrics(), I find "revenue" is available
    - I need to check what dimensions are available for revenue
    - Using get_dimensions(["revenue"]), I see "customer_name" is available
    - I need to check what entities are available
    - Using get_entities(["revenue"]), I confirm "customer" is an entity
    - I need quarterly data
    - I should order by revenue to see top customers
    - I should limit to top 10 results to verify the query works
    Parameters:
    metrics=["revenue"]
    group_by=["customer_name"]
    order_by=["revenue"]
    limit=10
    start_time="2024-01-01"
    end_time="2024-03-31"
</example>
<example>
Question: "What's our monthly user growth trend?"
    Thinking step-by-step:
    - I need to find user growth or new user metrics
    - Using list_metrics(), I find "new_users" is available
    - I need to check what dimensions are available
    - Using get_dimensions(["new_users"]), I see "ds" (date) is available
    - I need monthly grouping
    - I should order by date to show trend
    - Should get last 12 months to show trend
    Parameters:
    metrics=["new_users"]
    group_by=["ds__month"]
    order_by=["ds__month"]
    limit=12
    start_time="2023-01-01"
    end_time="2023-12-31"
</example>
</examples>

<parameters>
metrics: List of metric names to query for (required).
group_by: Optional list of dimensions to group by.
order_by: Optional list of fields to order by.
where: Optional WHERE clause to filter results.
limit: Optional limit for number of results.
start_time: Optional start time for the query (ISO format yyyy-mm-dd).
end_time: Optional end time for the query (ISO format yyyy-mm-dd).
</parameters>

<output>
Returns JSON containing query results including:
- Metric values
- Grouped dimensions/entities
- Calculated aggregations
- Query metadata (SQL, execution time, etc.)
</output> 