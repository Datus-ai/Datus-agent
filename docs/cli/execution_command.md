# Execution Command `!`

## 1. Overview

The Execution Command `!` provides direct command execution capabilities within the Datus-CLI environment. This command allows you to run system commands, database operations, and other executable tasks without leaving the interactive session.

## 2. Basic Usage

Execute commands by typing `!` followed by the command you want to run:

```bash
! ls -la
! python script.py
! git status
```

The execution command integrates seamlessly with the chat and context systems, allowing you to:

- Run database maintenance scripts
- Execute data processing workflows
- Perform system operations
- Launch external tools and utilities

### Examples

```bash
# Run a Python data processing script
! python data_processing.py --input data.csv --output results.csv

# Execute a shell script for data loading
! ./load_data.sh production_db

# Run database backup commands
! pg_dump mydb > backup.sql

# Execute system monitoring commands
! top -n 1
! df -h
```

## 3. Advanced Features

### Integration with Context

The execution command can leverage context from the `@` command system:

- Access environment variables set in the session
- Use file paths and database connections from context
- Execute scripts that operate on tables referenced in chat

### Session State

Commands executed with `!` maintain awareness of:

- Current working directory
- Environment variables
- Database connections established in the session
- Previous command outputs and error states

### Output Handling

The execution command provides:

- Real-time output streaming
- Error capturing and display
- Return code monitoring
- Integration with chat history for result reference

### Security Considerations

- Commands run with the same privileges as the Datus-CLI process
- Environment variables and secrets are handled securely
- Command history is maintained for audit purposes
- Interactive commands may require special handling

## Best Practices

1. **Use absolute paths** when referencing files to avoid confusion
2. **Test commands** in a safe environment before running in production
3. **Monitor output** for errors and unexpected results
4. **Combine with chat** to explain what commands do before executing them