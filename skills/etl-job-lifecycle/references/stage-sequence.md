# ETL Job Lifecycle Stage Sequence

Use this stage order unless the execution system imposes a stricter one:

1. DDL readiness
2. SQL generation or review
3. job submission
4. run monitoring
5. output validation
6. publish or rollback decision

Keep run-state reporting explicit:

- `submitted`
- `running`
- `succeeded`
- `failed`
- `cancelled`

Do not mark success from job completion alone. Success requires output validation to pass.

