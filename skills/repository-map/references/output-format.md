# Output Format

The repository map script returns JSON with:

- `root`
- `layer_counts`
- `layers`

## `layer_counts`

Counts of discovered SQL files per inferred layer.

## `layers`

Dictionary keyed by:

- `staging`
- `intermediate`
- `marts`
- `other`

Each entry is a list of file summaries with:

- `path`
- `output_table`
- `upstream_tables`
- `summary`

## Summary rules

Summaries are intentionally short and deterministic:

- `staging`: `staging transform from <sources>`
- `intermediate`: `intermediate model from <sources>`
- `marts`: `mart model from <sources>`
- `other`: `sql model from <sources>`

If there are more than three upstream sources, only the first three are named and the rest are summarized as `+N more`.

