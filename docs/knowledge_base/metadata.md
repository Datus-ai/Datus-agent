# Schema Metadata Intelligence

## Introduction

The metadata module is primarily used to enable LLMs to quickly match possible related table definition information and sample data based on user questions.

When you use the `bootstrap-kb` command, we initialize the SQL statements and sample data for creating tables/views/materialized views in the data source you specify into LanceDB.

This module contains two types of information: **table definition** and **sample data**.

## Data Structure of Table Definition

| Field Name | Explanation | Supported Database Types |
|------------|-------------|--------------------------|
| `catalog_name` | The top-level container in a database system. It typically represents a collection of databases and provides metadata about them, such as available schemas, tables, and security settings | StarRocks/Snowflake |
| `database_name` | A logical container that stores related data. It usually groups together multiple schemas and provides boundaries for data organization, security, and management. | DuckDB/MySQL/StarRocks/Snowflake |
| `schema_name` | A namespace inside a database. It organizes objects such as tables, views, functions, and procedures into logical groups. Schemas help avoid name conflicts and support role-based access. | DuckDB/Snowflake |
| `table_type` | The types of tables in the database, including `table`, `view`, and `mv` (abbreviation for materialized view). Each database supports table and view. DuckDB and Snowflake support materialized views. | All supported databases |
| `table_name` | Name of the table/view/materialized view | All supported databases |
| `definition` | SQL statements for creating tables/views/materialized views | All supported databases |
| `identifier` | The unique identifier of the current table, which is composed of `catalog_name`, `database_name`, `schema_name` and `table_name`. You don't need to worry about it, because you won't need it in most scenarios. | All supported databases |

## Data Structure of Sample Data

| Field Name | Explanation |
|------------|-------------|
| `catalog_name` | Same as above |
| `database_name` | Same as above |
| `schema_name` | Same as above |
| `table_type` | Same as above |
| `table_name` | Same as above |
| `sample_rows` | Sample data for the current table/view/mv. Usually it will be the first 5 items in the current table |
| `identifier` | Same as above |

## How to Build

You can build it using the `datus-agent bootstrap-kb` command:

```bash
datus-agent bootstrap-kb --namespace <your_namespace> --kb_update_strategy [check/overwrite/incremental]
```

### Command Line Parameter Description

- `--namespace`: The key corresponding to your database configuration
- `--kb_update_strategy`: Execution strategy, there are three options:
  - `check`: Check the number of data entries currently constructed
  - `overwrite`: Fully overwrite existing data
  - `incremental`: Incremental update: if existing data has changed, update it and append non-existent data

## Usage Examples

### Check Current Status
```bash
datus-agent bootstrap-kb --namespace my_database --kb_update_strategy check
```

### Full Rebuild
```bash
datus-agent bootstrap-kb --namespace my_database --kb_update_strategy overwrite
```

### Incremental Update
```bash
datus-agent bootstrap-kb --namespace my_database --kb_update_strategy incremental
```

## Best Practices

### Database Configuration
- Ensure your database namespace is properly configured in `agent.yml`
- Verify database connectivity before running bootstrap commands
- Use appropriate credentials with read access to system tables

### Update Strategy Selection
- Use `check` to verify current state without making changes
- Use `overwrite` for initial setup or when schema has changed significantly
- Use `incremental` for regular updates to capture new tables and changes

### Performance Considerations
- Large databases may take time to process during initial bootstrap
- Consider running during off-peak hours for production databases
- Monitor disk space as metadata is stored locally in LanceDB

## Troubleshooting

### Common Issues
- **Permission errors**: Ensure database user has access to system/information schema tables
- **Connection timeouts**: Check network connectivity and database availability
- **Large result sets**: Consider filtering to specific schemas if database is very large

### Verification
After bootstrap completion, verify the metadata was captured correctly:
- Check LanceDB storage directory for populated files
- Test search functionality through the CLI
- Verify sample data represents actual table contents