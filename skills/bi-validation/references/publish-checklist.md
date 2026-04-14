# BI Validation Checklist

After a BI publish:

1. confirm the target object exists
2. refresh or query the published object
3. compare a small set of key metrics against expected values or tolerances
4. report both absolute and relative differences when possible
5. block rollout when differences exceed the agreed threshold

Keep the metric set intentionally small. This is a publish gate, not a full analytical QA suite.
