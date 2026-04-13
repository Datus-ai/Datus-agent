# BI Publish Verification Checklist

After a BI publish:

1. Confirm the publish succeeded.
2. Refresh the target object or query its materialized result.
3. Compare a small set of key metrics against expected values or tolerance ranges.
4. Report both absolute and relative differences when possible.
5. Block rollout when differences exceed the agreed threshold.

Keep the metric set intentionally small. This skill is a publish gate, not a full analytical QA suite.

