# BI Validation Checklist

After a BI publish:

1. Call `get_dashboard` to confirm the dashboard exists and retrieve the full chart list
2. For **every** chart on the dashboard:
   a. Call `get_chart` to inspect configuration (chart_type, metrics, x_axis, dimensions, dataset_id)
   b. Verify each configuration field matches the intended design
   c. If `get_chart_data` is supported on the active platform, call it to confirm the chart returns data without errors
   d. Verify key numeric values match expected results or tolerances when those expectations are available
   e. If `get_chart_data` is not supported, record the data check as unsupported / N/A and rely on configuration inspection plus reference SQL or known expectations
3. If any chart fails validation:
   a. use `update_chart` only for fields the tool actually supports
   b. recreate the chart/panel when the issue is `dimensions`, `dataset_id`, or another unsupported wiring change
4. Report both absolute and relative differences when possible
5. Block rollout when any chart fails configuration or data validation

Do NOT skip any charts. Every chart on the dashboard must be validated. `get_chart_data` is required for every chart only on platforms that actually expose it.
