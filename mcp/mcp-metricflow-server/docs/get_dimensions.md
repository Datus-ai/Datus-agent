<instructions>
Get the dimensions for specified metrics using MetricFlow CLI

Dimensions are the attributes, features, or characteristics
that describe or categorize data. They are used to group and
filter metrics in queries.

This tool executes the `mf list dimensions --json` command locally
with optional metric filtering to return available dimensions.
</instructions>

<parameters>
metrics: Optional list of metric names to get dimensions for.
        If not provided, returns dimensions for all metrics.
</parameters>

<examples>
- "What dimensions can I use with the revenue metric?"
- "Show me all customer-related dimensions"
- "What attributes can I group sales data by?"
</examples>

<output>
Returns JSON containing dimension definitions including:
- Dimension names
- Data types
- Descriptions
- Available time grains (for time dimensions)
</output> 