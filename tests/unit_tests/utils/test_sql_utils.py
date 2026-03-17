from unittest.mock import patch

import pytest

from datus.tools.db_tools import connector_registry
from datus.utils.constants import DBType, SQLType
from datus.utils.json_utils import llm_result2json
from datus.utils.sql_utils import (
    _fallback_sql_type,
    _first_statement,
    _is_escaped,
    _match_dollar_tag,
    _metadata_pattern,
    extract_table_names,
    format_sql_to_pretty,
    metadata_identifier,
    normalize_sql,
    parse_context_switch,
    parse_dialect,
    parse_metadata_from_ddl,
    parse_read_dialect,
    parse_sql_type,
    parse_table_name_parts,
    parse_table_names_parts,
    strip_sql_comments,
)


@pytest.fixture(autouse=True)
def _register_test_capabilities():
    """Register capabilities for dialects used in tests."""
    connector_registry.register_handlers("mysql", capabilities={"database"})
    connector_registry.register_handlers("starrocks", capabilities={"catalog", "database"})
    connector_registry.register_handlers("oracle", capabilities={"database", "schema"})
    connector_registry.register_handlers("postgresql", capabilities={"database", "schema"})
    connector_registry.register_handlers("snowflake", capabilities={"catalog", "database", "schema"})
    yield


SQL = """create or replace TABLE GT.GT2.VARIANTS (
    "reference_name" VARCHAR(16777216),
    "start" NUMBER(38,0),
    "end" NUMBER(38,0),
    "reference_bases" VARCHAR(16777216),
    "alternate_bases" VARIANT,
    "quality" FLOAT,
    "filter" VARIANT,
    "names" VARIANT,
    "call" VARIANT,
    AA VARCHAR(16777216),
    AC VARIANT,
    AC1 NUMBER(38,0),
    AF FLOAT,
    AF1 FLOAT,
    AFR_AF FLOAT,
    AMR_AF FLOAT,
    AN NUMBER(38,0),
    ASN_AF FLOAT,
    AVGPOST FLOAT,
    CIEND VARIANT,
    CIPOS VARIANT,
    DP NUMBER(38,0),
    DP4 VARIANT,
    ERATE FLOAT,
    EUR_AF FLOAT,
    "fq" FLOAT,
    G3 VARIANT,
    HOMLEN NUMBER(38,0),
    HOMSEQ VARCHAR(16777216),
    HWE FLOAT,
    LDAF FLOAT,
    MQ NUMBER(38,0),
    PV4 VARIANT,
    RSQ FLOAT,
    SNPSOURCE VARIANT,
    SOURCE VARIANT,
    SVLEN NUMBER(38,0),
    SVTYPE VARCHAR(16777216),
    THETA FLOAT COMMENT 'This column contains the theta value for the variant.',
    VT VARCHAR(16777216) COMMENT 'This column contains the variant type for the variant.'
) COMMENT 'This table contains variant information for the reference genome.'; """


def test_parse_sql():
    table_meta = parse_metadata_from_ddl(SQL, "snowflake")
    print(table_meta)
    assert table_meta["table"]["name"] == "VARIANTS"
    assert table_meta["columns"][0]["name"] == "reference_name"
    assert len(table_meta["columns"]) == 40


def test_parse_oracle():
    table_meta = parse_metadata_from_ddl(
        """CREATE TABLE "attendance" (
    "link_to_event" NVARCHAR2(512) NOT NULL,
    "link_to_member" NVARCHAR2(512) NOT NULL,
    PRIMARY KEY ("link_to_event", "link_to_member")
);""",
        "oracle",
    )
    assert table_meta["table"]["name"] == "attendance"
    assert table_meta["columns"][0]["name"] == "link_to_event"
    assert len(table_meta["columns"]) == 2


def test_parse_mysql():
    table_meta = parse_metadata_from_ddl(
        """CREATE TABLE `connected` (
`atom_id` varchar(256) NOT NULL,
`atom_id2` varchar(256) NOT NULL,
`bond_id` varchar(256) NULL,
  PRIMARY KEY (`atom_id`, `atom_id2`),
  FOREIGN KEY (`atom_id`) REFERENCES `atom`(`atom_id`),
  FOREIGN KEY (`atom_id2`) REFERENCES `atom`(`atom_id`),
  FOREIGN KEY (`bond_id`) REFERENCES `bond`(`bond_id`)
);""",
        "mysql",
    )
    assert table_meta["table"]["name"] == "connected"
    assert table_meta["columns"][0]["name"] == "atom_id"
    assert len(table_meta["columns"]) == 3


def test_parse_postgresql():
    table_meta = parse_metadata_from_ddl(
        """CREATE TABLE "trans" (
trans_id bigint NOT NULL DEFAULT '0'::bigint,
account_id bigint NULL DEFAULT '0'::bigint,
date date NULL,
type text NULL,
operation text NULL,
amount bigint NULL,
balance bigint NULL,
k_symbol text NULL,
bank text NULL,
account bigint NULL,
    PRIMARY KEY (trans_id),
    FOREIGN KEY (account_id) REFERENCES account(account_id)
);""",
        "postgres",
    )
    assert table_meta["table"]["name"] == "trans"
    assert table_meta["columns"][0]["name"] == "trans_id"
    assert len(table_meta["columns"]) == 10


def test_parse_sqlserver():
    table_meta = parse_metadata_from_ddl(
        """CREATE TABLE [schools] (
[CDSCode] nvarchar(256) NOT NULL,
[NCESDist] nvarchar(MAX) NULL,
[NCESSchool] nvarchar(MAX) NULL,
[StatusType] nvarchar(MAX) NOT NULL,
[County] nvarchar(MAX) NOT NULL,
[District] nvarchar(MAX) NOT NULL,
[School] nvarchar(MAX) NULL,
[Street] nvarchar(MAX) NULL,
[StreetAbr] nvarchar(MAX) NULL,
[City] nvarchar(MAX) NULL,
[Zip] nvarchar(MAX) NULL,
[State] nvarchar(MAX) NULL,
[MailStreet] nvarchar(MAX) NULL,
[MailStrAbr] nvarchar(MAX) NULL,
[MailCity] nvarchar(MAX) NULL,
[MailZip] nvarchar(MAX) NULL,
[MailState] nvarchar(MAX) NULL,
[Phone] nvarchar(MAX) NULL,
[Ext] nvarchar(MAX) NULL,
[Website] nvarchar(MAX) NULL,
[OpenDate] date NULL,
[ClosedDate] date NULL,
[Charter] int NULL,
[CharterNum] nvarchar(MAX) NULL,
[FundingType] nvarchar(MAX) NULL,
[DOC] nvarchar(MAX) NOT NULL,
[DOCType] nvarchar(MAX) NOT NULL,
[SOC] nvarchar(MAX) NULL,
[SOCType] nvarchar(MAX) NULL,
[EdOpsCode] nvarchar(MAX) NULL,
[EdOpsName] nvarchar(MAX) NULL,
[EILCode] nvarchar(MAX) NULL,
[EILName] nvarchar(MAX) NULL,
[GSoffered] nvarchar(MAX) NULL,
[GSserved] nvarchar(MAX) NULL,
[Virtual] nvarchar(MAX) NULL,
[Magnet] int NULL,
[Latitude] float NULL,
[Longitude] float NULL,
[AdmFName1] nvarchar(MAX) NULL,
[AdmLName1] nvarchar(MAX) NULL,
[AdmEmail1] nvarchar(MAX) NULL,
[AdmFName2] nvarchar(MAX) NULL,
[AdmLName2] nvarchar(MAX) NULL,
[AdmEmail2] nvarchar(MAX) NULL,
[AdmFName3] nvarchar(MAX) NULL,
[AdmLName3] nvarchar(MAX) NULL,
[AdmEmail3] nvarchar(MAX) NULL,
[LastUpdate] date NOT NULL,
  PRIMARY KEY ([CDSCode])
);""",
        "sqlserver",
    )
    assert table_meta["table"]["name"] == "schools"
    assert table_meta["columns"][0]["name"] == "CDSCode"
    assert len(table_meta["columns"]) == 49


def test_json_utils():
    print(
        llm_result2json(
            """```json
[
  {
    "table": "NOAA_HISTORIC_SEVERE_STORMS.HAIL_REPORTS",
    "score": 0.9,
    "reasons": ["contains hail storm event data", "has 'timestamp' field for time range filtering",
    "has 'latitude' and 'longitude' fields for location data"]
  },
  {
    "table": "GEO_US_BOUNDARIES.ZIP_CODES",
    "score": 0.8,
    "reasons": ["contains zip code information",
    "has 'internal_point_lat' and 'internal_point_lon' fields for location data"]
  },
  {
    "table": "NOAA_HISTORIC_SEVERE_STORMS.STORMS_2020",
    "score": 0.7,
    "reasons": ["contains storm event data", "has 'event_begin_time' field for time range filtering",
    "has 'event_latitude' and 'event_longitude' fields for location data"]
  },
  {
    "table": "NOAA_HISTORIC_SEVERE_STORMS.STORMS_2019",
    "score": 0.7,
    "reasons": ["contains storm event data", "has 'event_begin_time' field for time range filtering",
    "has 'event_latitude' and 'event_longitude' fields for location data"]
  },
  {
    "table": "NOAA_HISTORIC_SEVERE_STORMS.STORMS_2018",
    "score": 0.7,
    "reasons": ["contains storm event data", "has 'event_begin_time' field for time range filtering",
    "has 'event_latitude' and 'event_longitude' fields for location data"]
  }
]
```

### Unmatched Tables with Reasons:
```json
[
  {
    "table": "NOAA_PRELIMINARY_SEVERE_STORMS.HAIL_REPORTS",
    "reason": "excluded as per user request to not use data from hail reports table"
  },
  {
    "table": "NOAA_SIGNIFICANT_EARTHQUAKES.EARTHQUAKES",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "NOAA_PRELIMINARY_SEVERE_STORMS.TORNADO_REPORTS",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "NOAA_PRELIMINARY_SEVERE_STORMS.WIND_REPORTS",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "NOAA_PASSIVE_BIOACOUSTIC.NCEI_AFSC_PAD_METADATA",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "NOAA_PASSIVE_BIOACOUSTIC.NCEI_NEFSC_PAD_METADATA",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "NOAA_PASSIVE_BIOACOUSTIC.NCEI_NRS_PAD_METADATA",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "NOAA_PASSIVE_BIOACOUSTIC.NCEI_SANCTSOUND_PAD_METADATA",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "NOAA_TSUNAMI.HISTORICAL_RUNUPS",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "NOAA_TSUNAMI.HISTORICAL_SOURCE_EVENT",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "NOAA_HURRICANES.HURRICANES",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "NOAA_ICOADS.ICOADS_CORE_2010",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "NOAA_ICOADS.ICOADS_CORE_2011",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "NOAA_ICOADS.ICOADS_CORE_2009",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "NOAA_ICOADS.ICOADS_CORE_2017",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "NOAA_ICOADS.ICOADS_CORE_2014",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "NOAA_ICOADS.ICOADS_CORE_2015",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "NOAA_ICOADS.ICOADS_CORE_2016",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "NOAA_ICOADS.ICOADS_CORE_2005",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "NOAA_ICOADS.ICOADS_CORE_2012",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "NOAA_ICOADS.ICOADS_CORE_1662_2000",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "NOAA_ICOADS.ICOADS_CORE_2006",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "NOAA_ICOADS.ICOADS_CORE_2008",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "NOAA_ICOADS.ICOADS_CORE_2013",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "NOAA_ICOADS.ICOADS_CORE_2001_2004",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "NOAA_ICOADS.ICOADS_CORE_2007",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "GEO_US_BOUNDARIES.NWS_FORECAST_REGIONS",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "GEO_US_BOUNDARIES.METROPOLITAN_DIVISIONS",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "GEO_US_BOUNDARIES.COUNTIES",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "GEO_US_BOUNDARIES.NATIONAL_OUTLINE",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "GEO_US_BOUNDARIES.URBAN_AREAS",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "GEO_US_BOUNDARIES.STATES",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "GEO_US_BOUNDARIES.ADJACENT_COUNTIES",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "GEO_US_BOUNDARIES.ADJACENT_STATES",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "GEO_US_BOUNDARIES.CNECTA",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "GEO_US_BOUNDARIES.COASTLINE",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "GEO_US_BOUNDARIES.CONGRESS_DISTRICT_115",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "GEO_US_BOUNDARIES.RAILWAYS",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "GEO_US_BOUNDARIES.CONGRESS_DISTRICT_116",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "GEO_US_BOUNDARIES.CBSA",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "GEO_US_BOUNDARIES.CSA",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "NOAA_PIFSC_METADATA.DCLDE_2020_1705_VISUAL_SIGHTINGS",
    "reason": "not relevant to hail storm events"
  },
  {
    "table": "NOAA_PIFSC_METADATA.DCLDE_2020_1706_VISUAL_SIGHTINGS",
    "reason": "not relevant to hail storm events"
  }]
  ```
"""
        )
    )


def parse_and_print(select_sql, except_tables, dialect=DBType.SQLITE):
    tables = extract_table_names(select_sql, dialect, ignore_empty=True)
    for table in tables:
        print(f"  - {table}")
    assert set(tables) == set(except_tables)


def test_parse_by_query():
    # Example SQL statements
    example_sql = """
    SELECT T2.CustomerID, SUM(T2.Price / T2.Amount), T1.Currency FROM customers AS T1 INNER JOIN transactions_1k AS T2
    ON T1.CustomerID = T2.CustomerID
    WHERE T2.CustomerID = ( SELECT CustomerID FROM yearmonth ORDER BY Consumption DESC LIMIT 1)
    GROUP BY T2.CustomerID, T1.Currency
    """
    parse_and_print(example_sql, ["customers", "transactions_1k", "yearmonth"])

    print("-" * 100)
    parse_and_print(
        """SELECT
  genex."case_barcode" AS "case_barcode",
  genex."sample_barcode" AS "sample_barcode",
  genex."aliquot_barcode" AS "aliquot_barcode",
  genex."HGNC_gene_symbol" AS "HGNC_gene_symbol",
  clinical_info."Variant_Type" AS "Variant_Type",
  genex."gene_id" AS "gene_id",
  genex."normalized_count" AS "normalized_count",
  genex."project_short_name" AS "project_short_name",
  clinical_info."demo__gender" AS "gender",
  clinical_info."demo__vital_status" AS "vital_status",
  clinical_info."demo__days_to_death" AS "days_to_death"
FROM (
  SELECT
    case_list."Variant_Type" AS "Variant_Type",
    case_list."case_barcode" AS "case_barcode",
    clinical."demo__gender",
    clinical."demo__vital_status",
    clinical."demo__days_to_death"
  FROM
    (SELECT
      mutation."case_barcode",
      mutation."Variant_Type"
    FROM
      "TCGA"."TCGA_VERSIONED"."SOMATIC_MUTATION_HG19_DCC_2017_02" AS mutation
    WHERE
      mutation."Hugo_Symbol" = 'CDKN2A'
      AND mutation."project_short_name" = 'TCGA-BLCA'
    GROUP BY
      mutation."case_barcode",
      mutation."Variant_Type"
    ORDER BY
      mutation."case_barcode"
    ) AS case_list /* end case_list */
  INNER JOIN
    "TCGA"."TCGA_VERSIONED"."CLINICAL_GDC_R39" AS clinical
  ON
    case_list."case_barcode" = clinical."submitter_id" /* end clinical annotation */ ) AS clinical_info
INNER JOIN
  "TCGA"."TCGA_VERSIONED"."RNASEQ_HG19_GDC_2017_02" AS genex
ON
  genex."case_barcode" = clinical_info."case_barcode"
WHERE
  genex."HGNC_gene_symbol" IN ('MDM2', 'TP53', 'CDKN1A','CCNE1')
ORDER BY
  "case_barcode",
  "HGNC_gene_symbol";
""",
        [
            "TCGA.TCGA_VERSIONED.SOMATIC_MUTATION_HG19_DCC_2017_02",
            "TCGA.TCGA_VERSIONED.CLINICAL_GDC_R39",
            "TCGA.TCGA_VERSIONED.RNASEQ_HG19_GDC_2017_02",
        ],
        dialect="snowflake",
    )

    print("-" * 100)
    parse_and_print(
        """SELECT account_id, MAX(payments) AS max_payment, MIN(payments) AS min_payment
        FROM loan GROUP BY account_id HAVING COUNT(account_id) > 1 AND (MAX(payments) - MIN(payments)) > 2;
        WITH cte AS (SELECT * FROM loan)
        SELECT * FROM cte;
        """,
        ["loan"],
        dialect="postgres",
    )


def test_parse_duckdb():
    table_meta = parse_metadata_from_ddl(
        """CREATE TABLE abc.test (
id bigint primary key,
account_id bigint null default '0',
date date null,
type text null)""",
        dialect=DBType.DUCKDB,
    )
    print(table_meta)


def test_parse_sqlite():
    table_meta = parse_metadata_from_ddl(
        """CREATE TABLE date (
          d_datekey          INT,     -- identifier, unique id -- e.g. 19980327 (what we use)
          d_date             TEXT,  -- varchar(18), --fixed text, size 18, longest: december 22, 1998
          d_dayofweek        TEXT,  -- varchar(8), --fixed text, size 8, sunday, monday, ..., saturday)
          d_month            TEXT,  -- varchar(9), --fixed text, size 9: january, ..., december
          d_year             INT,     -- unique value 1992-1998
          d_yearmonthnum     INT,     -- numeric (yyyymm) -- e.g. 199803
          d_yearmonth        TEXT,  -- varchar(7), --fixed text, size 7: mar1998 for example
          d_daynuminweek     INT,     -- numeric 1-7
          d_daynuminmonth    INT,     -- numeric 1-31
          d_daynuminyear     INT,     -- numeric 1-366
          d_monthnuminyear   INT,     -- numeric 1-12
          d_weeknuminyear    INT,     -- numeric 1-53
          d_sellingseason    TEXT,  -- varchar(12), --text, size 12 (christmas, summer,...)
          d_lastdayinweekfl  INT,     -- 1 bit
          d_lastdayinmonthfl INT,     -- 1 bit
          d_holidayfl        INT,     -- 1 bit
          d_weekdayfl        INT,     -- 1 bit
          PRIMARY KEY (d_datekey)
          )""",
        dialect=DBType.SQLITE,
    )

    print(table_meta)
    tb_info = table_meta["table"]
    assert tb_info["name"] == "date"
    assert tb_info["database_name"] == ""
    assert tb_info["schema_name"] == ""
    assert table_meta["columns"][0]["name"] == "d_datekey"
    assert len(table_meta["columns"]) == 17


def test_parse_sqlite_select():
    sql = """WITH SubQuery AS (SELECT DISTINCT T1.atom_id, T1.element, T1.molecule_id, T2.label
    FROM atom AS T1 INNER JOIN molecule AS T2 ON T1.molecule_id = T2.molecule_id WHERE T2.molecule_id = 'TR006')
    SELECT CAST(COUNT(CASE WHEN element = 'h' THEN atom_id ELSE NULL END) AS REAL) / (CASE WHEN COUNT(atom_id) = 0
    THEN NULL ELSE COUNT(atom_id) END) AS ratio, label FROM SubQuery GROUP BY label"""
    tables = extract_table_names(sql, dialect=DBType.SQLITE)
    assert set(tables) == {"atom", "molecule"}


def test_extract_table_names():
    assert set(extract_table_names("SELECT * FROM default_catalog.bar.baz")) == {"default_catalog.bar.baz"}
    sql_three_part = "SELECT * FROM foo.bar.baz"
    sql_two_part = "SELECT * FROM foo.bar"
    assert set(extract_table_names(sql_three_part, dialect="sqlserver", ignore_empty=True)) == {"foo.bar.baz"}
    for dialect in ["sqlserver", "postgresql", "mysql", "starrocks"]:
        assert set(extract_table_names(sql_two_part, dialect=dialect, ignore_empty=True)) == {"foo.bar"}


def test_parse_full_tables():
    table_meta = parse_table_name_parts("test.abc", dialect=DBType.DUCKDB)
    assert table_meta["schema_name"] == "test"
    assert table_meta["table_name"] == "abc"
    assert table_meta["database_name"] == ""
    assert table_meta["catalog_name"] == ""

    table_meta = parse_table_name_parts("`test`.abc", dialect="mysql")
    assert table_meta["database_name"] == "test"
    assert table_meta["table_name"] == "abc"
    assert table_meta["schema_name"] == ""
    assert table_meta["catalog_name"] == ""

    table_meta = parse_table_name_parts('''TEST_DB."test_schema"."abc"''', dialect="snowflake")
    assert table_meta["schema_name"] == "test_schema"
    assert table_meta["table_name"] == "abc"
    assert table_meta["database_name"] == "TEST_DB"
    assert table_meta["catalog_name"] == ""


def test_parse_sql_type():
    sql = """---Basic statistics and correlation
SELECT
    COUNT(*) as total_records,
    MIN(time) as start_time,
    MAX(time) as end_time,
    AVG(gold) as avg_gold,
    AVG(bitcoin) as avg_bitcoin,
    STDDEV(gold) as std_gold,
    STDDEV(bitcoin) as std_bitcoin,
    CORR(gold, bitcoin) as correlation,
    COVAR_POP(gold, bitcoin) as covariance,
    POWER(CORR(gold, bitcoin), 2) as r_squared
FROM gold_vs_bitcoin"""
    assert parse_sql_type(sql, dialect=DBType.DUCKDB) == SQLType.SELECT
    assert (
        parse_sql_type("show create table `default_catalog`.`ac_manage`.`v_udata_ac_info`", dialect="starrocks")
        == SQLType.METADATA_SHOW
    )

    assert (
        parse_sql_type("select * from `default_catalog`.`ac_manage`.`v_udata_ac_info`", dialect="starrocks")
        == SQLType.SELECT
    )

    assert parse_sql_type("   ", dialect=DBType.DUCKDB) == SQLType.UNKNOWN

    merge_sql = (
        "MERGE INTO target USING source ON target.id = source.id WHEN MATCHED THEN UPDATE SET value = source.value"
    )
    assert parse_sql_type(merge_sql, dialect="snowflake") == SQLType.MERGE

    assert parse_sql_type("EXPLAIN SELECT * FROM gold_vs_bitcoin", dialect=DBType.DUCKDB) == SQLType.EXPLAIN

    assert parse_sql_type("SHOW TABLES", dialect=DBType.DUCKDB) == SQLType.METADATA_SHOW

    assert parse_sql_type("SHOW CATALOGS", dialect="starrocks") == SQLType.METADATA_SHOW

    assert parse_sql_type("USE test", dialect=DBType.DUCKDB) == SQLType.CONTENT_SET
    assert parse_sql_type("USE test", dialect="mysql") == SQLType.CONTENT_SET
    assert parse_sql_type("USE test", dialect="starrocks") == SQLType.CONTENT_SET
    assert parse_sql_type(" USE test ", dialect="snowflake") == SQLType.CONTENT_SET


def test_parse_sql_type_with():
    sql = """WITH hourly_data AS (
        SELECT
            EXTRACT(HOUR FROM time) as hour_of_day,
            AVG(gold) as avg_gold,
            AVG(bitcoin) as avg_bitcoin,
            CORR(gold, bitcoin) as hourly_correlation
        FROM gold_vs_bitcoin
        GROUP BY EXTRACT(HOUR FROM time)
    ),
                  rolling_corr AS (
                      SELECT
                 time,
                 CORR(gold, bitcoin) OVER (
                 ORDER BY time
                 ROWS BETWEEN 50 PRECEDING AND CURRENT ROW
                 ) as rolling_correlation_50
             FROM gold_vs_bitcoin
                 )
    SELECT
        'Hourly Analysis' as analysis_type,
        hour_of_day,
        hourly_correlation
    FROM hourly_data
    UNION ALL
    SELECT
        'Rolling Correlation' as analysis_type,
        NULL as hour_of_day,
        AVG(rolling_correlation_50) as hourly_correlation
    FROM rolling_corr
    WHERE rolling_correlation_50 IS NOT NULL;"""
    sql_type = parse_sql_type(sql, dialect=DBType.DUCKDB)
    assert sql_type == SQLType.SELECT

    sql = """with round_user as (
              select dtstatdate,
                     case
                         when mode in (401,402,403) then 'FIRST_PERSON'
                         when mode =101 then 'solo'
                         when mode =102 then 'double-row'
                         when mode in (103,603) then 'four-row'
                         end modename,
                     vplayerid,
                     sum(roundcnt) roundcnt,
                     sum(roundtime) roundtime
              from dws_jordass_mode_roundrecord_di
              where ((dtstatdate between '20240326' and '20240409')
                  or (dtstatdate between '20240528' and '20240611'))
                and mode in (401,402,403,101,102,103,603)
              group by dtstatdate,
                       case when mode in (401,402,403) then 'FIRST_PERSON'
                            when mode =101 then 'solo'
                            when mode =102 then 'double-row'
                            when mode in (103,603) then 'four-row'
                           end,
                       vplayerid
          )

          select
              a.dtstatdate,
              a.modename,
              count(distinct a.vplayerid) iusernum,
              sum(a.roundcnt) roundcnt,
              sum(a.roundtime) roundtime,
              count(distinct b1.vplayerid) stay2,
              count(distinct b2.vplayerid) stay7,
              count(distinct c1.vplayerid) playstay2,
              count(distinct c2.vplayerid) playstay7
          from (
                   select * from round_user
                   where((dtstatdate between '20240326' and '20240403')
                       or (dtstatdate between '20240528' and '20240605'))
               ) a
                   left join (
              select dtstatdate,vplayerid
              from dws_jordass_login_di
              where ((dtstatdate between '20240326' and '20240404')
                  or (dtstatdate between '20240528' and '20240606'))
                and platid =255
              group by dtstatdate,vplayerid
          ) b1
                             on a.vplayerid = b1.vplayerid and date_add(a.dtstatdate,1) = b1.dtstatdate
                   left join (
              select dtstatdate,vplayerid
              from dws_jordass_login_di
              where ((dtstatdate between '20240326' and '20240409') or (dtstatdate between '20240528' and '20240611'))
                and platid =255
              group by dtstatdate,vplayerid
          ) b2
                             on a.vplayerid = b2.vplayerid and date_add(a.dtstatdate,6) = b2.dtstatdate
                   left join round_user c1
                             on a.vplayerid = c1.vplayerid and date_add(a.dtstatdate,1) = c1.dtstatdate
                                 and a.modename= c1.modename
                   left join round_user c2
                             on a.vplayerid = c2.vplayerid and date_add(a.dtstatdate,6) = c2.dtstatdate
                                 and a.modename= c2.modename
          group by a.dtstatdate,a.modename
    """
    sql_type = parse_sql_type(sql, dialect="starrocks")
    assert sql_type == SQLType.SELECT

    sql_type = parse_sql_type(
        """WITH action_films AS (
        SELECT
            f.title,
            f.length
        FROM
            film f
                INNER JOIN film_category fc USING (film_id)
                INNER JOIN category c USING(category_id)
        WHERE
            c.name = 'Action'
    )
    SELECT * FROM action_films;""",
        dialect="postgresql",
    )
    assert sql_type == SQLType.SELECT


def test_parse_sql_type_union_statement():
    sql = "SELECT 1 UNION SELECT 2"
    assert parse_sql_type(sql, dialect=DBType.DUCKDB) == SQLType.SELECT


def test_parse_sql_type_wrapped_select():
    sql = "(WITH cte AS (SELECT 1) SELECT * FROM cte)"
    assert parse_sql_type(sql, dialect=DBType.DUCKDB) == SQLType.SELECT


def test_parse_context_switch_duckdb():
    # simple and fuzzy
    result = parse_context_switch("USE analytics;", dialect=DBType.DUCKDB)
    assert result == {
        "command": "USE",
        "target": "schema",
        "catalog_name": "",
        "database_name": "",
        "schema_name": "analytics",
        "fuzzy": True,
        "raw": "USE analytics",
    }
    # full
    result = parse_context_switch("use sales.analytics", dialect=DBType.DUCKDB)
    assert result == {
        "command": "USE",
        "target": "schema",
        "catalog_name": "",
        "database_name": "sales",
        "schema_name": "analytics",
        "fuzzy": False,
        "raw": "use sales.analytics",
    }


def test_parse_context_switch_mysql():
    result = parse_context_switch("USE `orders`", dialect="mysql")
    assert result == {
        "command": "USE",
        "target": "database",
        "catalog_name": "",
        "database_name": "orders",
        "schema_name": "",
        "fuzzy": False,
        "raw": "USE `orders`",
    }

    result = parse_context_switch("USE orders", dialect="mysql")
    assert result == {
        "command": "USE",
        "target": "database",
        "catalog_name": "",
        "database_name": "orders",
        "schema_name": "",
        "fuzzy": False,
        "raw": "USE orders",
    }


def test_parse_context_switch_starrocks():
    # set_catalog
    result = parse_context_switch("SET catalog lakehouse", dialect="starrocks")
    assert result == {
        "command": "SET",
        "target": "catalog",
        "catalog_name": "lakehouse",
        "database_name": "",
        "schema_name": "",
        "fuzzy": False,
        "raw": "SET catalog lakehouse",
    }

    # datalog.db
    result = parse_context_switch("USE lakehouse.sales", dialect="starrocks")
    assert result == {
        "command": "USE",
        "target": "database",
        "catalog_name": "lakehouse",
        "database_name": "sales",
        "schema_name": "",
        "fuzzy": False,
        "raw": "USE lakehouse.sales",
    }

    # db
    result = parse_context_switch("USE sales", dialect="starrocks")
    assert result == {
        "command": "USE",
        "target": "database",
        "catalog_name": "",
        "database_name": "sales",
        "schema_name": "",
        "fuzzy": False,
        "raw": "USE sales",
    }


def test_parse_context_switch_snowflake():
    result = parse_context_switch("USE DATABASE analytics", dialect="snowflake")
    assert result == {
        "command": "USE",
        "target": "database",
        "catalog_name": "",
        "database_name": "analytics",
        "schema_name": "",
        "fuzzy": False,
        "raw": "USE DATABASE analytics",
    }

    result = parse_context_switch("USE analytics", dialect="snowflake")
    assert result == {
        "command": "USE",
        "target": "database",
        "catalog_name": "",
        "database_name": "analytics",
        "schema_name": "",
        "fuzzy": False,
        "raw": "USE analytics",
    }

    # db.schema
    result = parse_context_switch("USE sales.analytics", dialect="snowflake")
    assert result == {
        "command": "USE",
        "target": "schema",
        "catalog_name": "",
        "database_name": "sales",
        "schema_name": "analytics",
        "fuzzy": False,
        "raw": "USE sales.analytics",
    }
    result = parse_context_switch("USE schema sales.analytics", dialect="snowflake")
    assert result == {
        "command": "USE",
        "target": "schema",
        "catalog_name": "",
        "database_name": "sales",
        "schema_name": "analytics",
        "fuzzy": False,
        "raw": "USE schema sales.analytics",
    }

    # schema
    result = parse_context_switch("USE schema analytics", dialect="snowflake")
    assert result == {
        "command": "USE",
        "target": "schema",
        "catalog_name": "",
        "database_name": "",
        "schema_name": "analytics",
        "fuzzy": False,
        "raw": "USE schema analytics",
    }


def test_first_statement():
    sql = "INSERT INTO t VALUES ('a;b'); SELECT 1;"
    assert _first_statement(sql) == "INSERT INTO t VALUES ('a;b')"

    sql = 'INSERT INTO t VALUES ("a;b"); SELECT 1;'
    assert _first_statement(sql) == 'INSERT INTO t VALUES ("a;b")'

    sql = "SELECT 1;"
    assert _first_statement(sql) == "SELECT 1"

    sql = "DO $$ BEGIN RAISE NOTICE 'foo;'; END $$; SELECT 1;"
    assert _first_statement(sql) == "DO $$ BEGIN RAISE NOTICE 'foo;'; END $$"


class TestMetadataIdentifier:
    def test_mysql(self):
        result = metadata_identifier(database_name="mydb", table_name="t1", dialect="mysql")
        assert result == "mydb.t1"

    def test_mysql_with_catalog(self):
        # MySQL has no catalog support; catalog_name is ignored
        result = metadata_identifier(catalog_name="cat", database_name="mydb", table_name="t1", dialect="mysql")
        assert result == "mydb.t1"

    def test_starrocks(self):
        result = metadata_identifier(database_name="db", table_name="t", dialect="starrocks")
        assert result == "db.t"

    def test_oracle(self):
        result = metadata_identifier(database_name="db", schema_name="hr", table_name="emp", dialect="oracle")
        assert result == "db.hr.emp"

    def test_postgresql(self):
        result = metadata_identifier(database_name="db", schema_name="public", table_name="t", dialect="postgresql")
        assert result == "db.public.t"

    def test_snowflake(self):
        result = metadata_identifier(
            catalog_name="wh", database_name="db", schema_name="s", table_name="t", dialect="snowflake"
        )
        assert result == "wh.db.s.t"

    def test_snowflake_no_catalog(self):
        result = metadata_identifier(database_name="db", schema_name="s", table_name="t", dialect="snowflake")
        assert result == "db.s.t"


def test_parse_table_name_parts_sqlite():
    """Cover line 212: sqlite returns ["database_name", "table_name"] mapping."""
    result = parse_table_name_parts("main.users", dialect=DBType.SQLITE)
    assert result["database_name"] == "main"
    assert result["table_name"] == "users"
    assert result.get("schema_name", "") == ""

    # Single-part table name
    result = parse_table_name_parts("users", dialect=DBType.SQLITE)
    assert result["table_name"] == "users"


def test_parse_sql_type_starrocks_metadata_none_parse():
    """Cover line 559: starrocks metadata pattern when sqlglot.parse_one returns None."""
    from unittest.mock import patch

    # sqlglot.parse_one returns None for certain unparseable starrocks metadata commands
    with patch("datus.utils.sql_utils.sqlglot.parse_one", return_value=None):
        result = parse_sql_type("SHOW CREATE TABLE db.tbl", dialect="starrocks")
        assert result == SQLType.METADATA_SHOW


def test_parse_context_switch_duckdb_set_schema():
    """Cover line 820: DuckDB SET SCHEMA without database sets fuzzy=False."""
    result = parse_context_switch("SET SCHEMA main", dialect=DBType.DUCKDB)
    assert result is not None
    assert result["command"] == "SET"
    assert result["target"] == "schema"
    assert result["schema_name"] == "main"
    assert result["database_name"] == ""
    assert result["fuzzy"] is False


@pytest.fixture(autouse=True)
def _register_test_capabilitiesExtended():
    """Register capabilities for dialects used in tests."""
    connector_registry.register_handlers("mysql", capabilities={"database"})
    connector_registry.register_handlers("starrocks", capabilities={"catalog", "database"})
    connector_registry.register_handlers("oracle", capabilities={"database", "schema"})
    connector_registry.register_handlers("postgresql", capabilities={"database", "schema"})
    connector_registry.register_handlers("snowflake", capabilities={"catalog", "database", "schema"})
    yield


# ---------------------------------------------------------------------------
# parse_read_dialect
# ---------------------------------------------------------------------------


class TestParseReadDialect:
    def test_postgres(self):
        assert parse_read_dialect("postgres") == "postgres"
        assert parse_read_dialect("postgresql") == "postgres"
        assert parse_read_dialect("redshift") == "postgres"
        assert parse_read_dialect("greenplum") == "postgres"

    def test_hive(self):
        assert parse_read_dialect("spark") == "hive"
        assert parse_read_dialect("databricks") == "hive"
        assert parse_read_dialect("hive") == "hive"
        assert parse_read_dialect("starrocks") == "hive"

    def test_tsql(self):
        assert parse_read_dialect("mssql") == "tsql"
        assert parse_read_dialect("sqlserver") == "tsql"

    def test_passthrough(self):
        assert parse_read_dialect("snowflake") == "snowflake"
        assert parse_read_dialect("mysql") == "mysql"

    def test_empty_string(self):
        assert parse_read_dialect("") == ""

    def test_whitespace_trimmed(self):
        assert parse_read_dialect("  postgres  ") == "postgres"


# ---------------------------------------------------------------------------
# parse_dialect
# ---------------------------------------------------------------------------


class TestParseDialect:
    def test_postgres(self):
        assert parse_dialect("postgres") == "postgres"
        assert parse_dialect("postgresql") == "postgres"

    def test_tsql(self):
        assert parse_dialect("mssql") == "tsql"
        assert parse_dialect("sqlserver") == "tsql"

    def test_passthrough(self):
        assert parse_dialect("snowflake") == "snowflake"
        assert parse_dialect("mysql") == "mysql"
        assert parse_dialect("duckdb") == "duckdb"


# ---------------------------------------------------------------------------
# strip_sql_comments
# ---------------------------------------------------------------------------


class TestStripSqlComments:
    def test_removes_block_comments(self):
        sql = "SELECT /* this is a comment */ 1"
        result = strip_sql_comments(sql)
        assert "comment" not in result
        assert "SELECT" in result

    def test_removes_line_comments(self):
        sql = "SELECT 1 -- this is a comment\nFROM t"
        result = strip_sql_comments(sql)
        assert "comment" not in result
        assert "FROM t" in result

    def test_multiline_block_comment(self):
        sql = "SELECT /* multi\nline\ncomment */ 1"
        result = strip_sql_comments(sql)
        assert "multi" not in result


# ---------------------------------------------------------------------------
# _is_escaped
# ---------------------------------------------------------------------------


class TestIsEscaped:
    def test_not_escaped(self):
        assert _is_escaped("abc'def", 3) is False

    def test_escaped_by_one_backslash(self):
        assert _is_escaped("abc\\'def", 4) is True

    def test_escaped_by_two_backslashes_not_escaped(self):
        # Two backslashes before = not escaped (even number)
        assert _is_escaped("abc\\\\'def", 5) is False

    def test_index_at_start(self):
        assert _is_escaped("'test", 0) is False


# ---------------------------------------------------------------------------
# _match_dollar_tag
# ---------------------------------------------------------------------------


class TestMatchDollarTag:
    def test_matches_simple_tag(self):
        text = "$$hello$$"
        tag = _match_dollar_tag(text, 0)
        assert tag == "$$"

    def test_matches_named_tag(self):
        text = "$body$hello$body$"
        tag = _match_dollar_tag(text, 0)
        assert tag == "$body$"

    def test_no_match(self):
        text = "abc"
        assert _match_dollar_tag(text, 0) is None

    def test_matches_at_offset(self):
        text = "  $$content$$"
        assert _match_dollar_tag(text, 2) == "$$"


# ---------------------------------------------------------------------------
# _first_statement - extended cases
# ---------------------------------------------------------------------------


class TestFirstStatementExtended:
    def test_backtick_quotes(self):
        sql = "SELECT `col;name` FROM t; SELECT 2"
        result = _first_statement(sql)
        assert result == "SELECT `col;name` FROM t"

    def test_bracket_quotes(self):
        sql = "SELECT [col;name] FROM t; SELECT 2"
        result = _first_statement(sql)
        assert result == "SELECT [col;name] FROM t"

    def test_escaped_single_quote(self):
        sql = "INSERT INTO t VALUES ('it\\'s ok'); SELECT 1"
        result = _first_statement(sql)
        assert "INSERT" in result

    def test_double_single_quote_escape(self):
        sql = "INSERT INTO t VALUES ('it''s ok'); SELECT 1"
        result = _first_statement(sql)
        assert "INSERT INTO t VALUES" in result

    def test_double_double_quote_escape(self):
        sql = 'INSERT INTO t VALUES ("it""s ok"); SELECT 1'
        result = _first_statement(sql)
        assert "INSERT" in result

    def test_double_backtick_escape(self):
        sql = "SELECT `col``name` FROM t; SELECT 2"
        result = _first_statement(sql)
        assert "SELECT" in result

    def test_empty_sql(self):
        assert _first_statement("") == ""

    def test_no_semicolon(self):
        sql = "SELECT 1"
        assert _first_statement(sql) == "SELECT 1"


# ---------------------------------------------------------------------------
# _fallback_sql_type
# ---------------------------------------------------------------------------


class TestFallbackSqlType:
    def test_select(self):
        assert _fallback_sql_type("SELECT * FROM t") == SQLType.SELECT

    def test_insert(self):
        assert _fallback_sql_type("INSERT INTO t VALUES (1)") == SQLType.INSERT

    def test_update(self):
        assert _fallback_sql_type("UPDATE t SET x=1") == SQLType.UPDATE

    def test_delete(self):
        assert _fallback_sql_type("DELETE FROM t") == SQLType.DELETE

    def test_ddl_create(self):
        assert _fallback_sql_type("CREATE TABLE t (id INT)") == SQLType.DDL

    def test_ddl_drop(self):
        assert _fallback_sql_type("DROP TABLE t") == SQLType.DDL

    def test_ddl_alter(self):
        assert _fallback_sql_type("ALTER TABLE t ADD COLUMN c INT") == SQLType.DDL

    def test_show(self):
        assert _fallback_sql_type("SHOW TABLES") == SQLType.METADATA_SHOW

    def test_explain(self):
        assert _fallback_sql_type("EXPLAIN SELECT * FROM t") == SQLType.EXPLAIN

    def test_use(self):
        assert _fallback_sql_type("USE mydb") == SQLType.CONTENT_SET

    def test_with_select(self):
        result = _fallback_sql_type("WITH cte AS (SELECT 1) SELECT * FROM cte")
        assert result == SQLType.SELECT

    def test_with_insert(self):
        result = _fallback_sql_type("WITH cte AS (SELECT 1) INSERT INTO t SELECT * FROM cte")
        assert result == SQLType.INSERT

    def test_empty_string(self):
        assert _fallback_sql_type("") is None

    def test_unknown_keyword(self):
        assert _fallback_sql_type("FOOBAR * FROM t") is None

    def test_values_keyword(self):
        assert _fallback_sql_type("VALUES (1, 2, 3)") == SQLType.SELECT

    def test_replace_keyword(self):
        assert _fallback_sql_type("REPLACE INTO t VALUES (1)") == SQLType.INSERT


# ---------------------------------------------------------------------------
# _metadata_pattern
# ---------------------------------------------------------------------------


class TestMetadataPattern:
    def test_matches_show(self):
        pattern = _metadata_pattern()
        assert pattern.match("SHOW TABLES")

    def test_matches_describe(self):
        pattern = _metadata_pattern()
        assert pattern.match("DESCRIBE t")

    def test_matches_pragma(self):
        pattern = _metadata_pattern()
        assert pattern.match("PRAGMA table_info(t)")

    def test_no_match_select(self):
        pattern = _metadata_pattern()
        assert not pattern.match("SELECT 1")

    def test_singleton(self):
        p1 = _metadata_pattern()
        p2 = _metadata_pattern()
        assert p1 is p2


# ---------------------------------------------------------------------------
# parse_sql_type - additional cases
# ---------------------------------------------------------------------------


class TestParseSqlTypeExtended:
    def test_empty_string(self):
        assert parse_sql_type("", "duckdb") == SQLType.UNKNOWN

    def test_none_like_input(self):
        assert parse_sql_type(None, "duckdb") == SQLType.UNKNOWN

    def test_only_whitespace(self):
        assert parse_sql_type("   ", "duckdb") == SQLType.UNKNOWN

    def test_insert_statement(self):
        assert parse_sql_type("INSERT INTO t VALUES (1)", "duckdb") == SQLType.INSERT

    def test_update_statement(self):
        assert parse_sql_type("UPDATE t SET x = 1", "duckdb") == SQLType.UPDATE

    def test_delete_statement(self):
        assert parse_sql_type("DELETE FROM t WHERE id = 1", "duckdb") == SQLType.DELETE

    def test_create_table(self):
        assert parse_sql_type("CREATE TABLE t (id INT)", "duckdb") == SQLType.DDL

    def test_drop_table(self):
        assert parse_sql_type("DROP TABLE t", "duckdb") == SQLType.DDL

    def test_alter_table(self):
        assert parse_sql_type("ALTER TABLE t ADD COLUMN c INT", "duckdb") == SQLType.DDL

    def test_use_statement_mysql(self):
        assert parse_sql_type("USE mydb", "mysql") == SQLType.CONTENT_SET

    def test_describe_duckdb(self):
        result = parse_sql_type("DESCRIBE t", "duckdb")
        assert result == SQLType.METADATA_SHOW

    def test_pragma_sqlite(self):
        result = parse_sql_type("PRAGMA table_info(t)", DBType.SQLITE)
        assert result == SQLType.METADATA_SHOW

    def test_sql_with_comments(self):
        sql = "-- This is a comment\nSELECT * FROM t"
        assert parse_sql_type(sql, "duckdb") == SQLType.SELECT

    def test_fallback_when_parse_exception(self):
        with patch("datus.utils.sql_utils.sqlglot.parse_one", side_effect=Exception("parse error")):
            result = parse_sql_type("SELECT * FROM t", "duckdb")
        # Falls back to keyword-based detection
        assert result == SQLType.SELECT

    def test_starrocks_metadata_returns_none_from_parser(self):
        with patch("datus.utils.sql_utils.sqlglot.parse_one", return_value=None):
            result = parse_sql_type("SHOW DATABASES", "starrocks")
        assert result == SQLType.METADATA_SHOW


# ---------------------------------------------------------------------------
# normalize_sql
# ---------------------------------------------------------------------------


class TestNormalizeSql:
    def test_removes_newlines(self):
        sql = "SELECT *\nFROM t\nWHERE id = 1"
        result = normalize_sql(sql)
        assert "\n" not in result
        assert "SELECT * FROM t WHERE id = 1" == result

    def test_removes_tabs(self):
        sql = "SELECT\t*\tFROM\tt"
        result = normalize_sql(sql)
        assert "\t" not in result

    def test_collapses_spaces(self):
        sql = "SELECT   *   FROM   t"
        result = normalize_sql(sql)
        assert "  " not in result

    def test_strips_leading_trailing(self):
        sql = "  SELECT 1  "
        result = normalize_sql(sql)
        assert result == "SELECT 1"

    def test_mixed_whitespace(self):
        sql = "SELECT\r\n*\r\nFROM\r\nt"
        result = normalize_sql(sql)
        assert "\r" not in result
        assert "\n" not in result


# ---------------------------------------------------------------------------
# format_sql_to_pretty
# ---------------------------------------------------------------------------


class TestFormatSqlToPretty:
    def test_formats_valid_sql(self):
        sql = "SELECT * FROM t WHERE id=1"
        result = format_sql_to_pretty(sql, "duckdb")
        assert "SELECT" in result

    def test_empty_string_passthrough(self):
        assert format_sql_to_pretty("", "duckdb") == ""

    def test_none_passthrough(self):
        assert format_sql_to_pretty(None, "duckdb") is None

    def test_fallback_on_exception(self):
        sql = "SELECT * FROM t"
        with patch("datus.utils.sql_utils.sqlglot.transpile", side_effect=Exception("parse error")):
            result = format_sql_to_pretty(sql, "duckdb")
        assert result == sql

    def test_fallback_on_empty_transpile_result(self):
        sql = "SELECT 1"
        with patch("datus.utils.sql_utils.sqlglot.transpile", return_value=[]):
            result = format_sql_to_pretty(sql, "duckdb")
        assert result == sql


# ---------------------------------------------------------------------------
# parse_table_names_parts (plural)
# ---------------------------------------------------------------------------


class TestParseTableNamesParts:
    def test_single_table(self):
        result = parse_table_names_parts(["schema.table"], dialect=DBType.DUCKDB)
        assert len(result) == 1
        assert result[0]["table_name"] == "table"

    def test_multiple_tables(self):
        result = parse_table_names_parts(["db.schema.table1", "db.schema.table2"], dialect="snowflake")
        assert len(result) == 2

    def test_empty_list(self):
        result = parse_table_names_parts([])
        assert result == []


# ---------------------------------------------------------------------------
# parse_table_name_parts - extended
# ---------------------------------------------------------------------------


class TestParseTableNamePartsExtended:
    def test_mysql_single_part(self):
        result = parse_table_name_parts("mytable", dialect="mysql")
        assert result["table_name"] == "mytable"

    def test_duckdb_three_parts(self):
        result = parse_table_name_parts("mydb.myschema.mytable", dialect=DBType.DUCKDB)
        assert result["database_name"] == "mydb"
        assert result["schema_name"] == "myschema"
        assert result["table_name"] == "mytable"

    def test_empty_string(self):
        result = parse_table_name_parts("", dialect="snowflake")
        assert result["table_name"] == ""

    def test_bracket_quoted(self):
        result = parse_table_name_parts("[mydb].[schema].[table]", dialect="sqlserver")
        assert result["table_name"] == "table"

    def test_snowflake_four_parts_excess(self):
        # More parts than expected - takes last N
        result = parse_table_name_parts("extra.catalog.db.schema.table", dialect="snowflake")
        assert result["table_name"] == "table"

    def test_unknown_dialect_fallback(self):
        # Unknown dialect falls through to default behavior
        result = parse_table_name_parts("a.b.c.d", dialect="unknown_dialect_xyz")
        assert result["table_name"] == "d"


# ---------------------------------------------------------------------------
# parse_context_switch - additional cases
# ---------------------------------------------------------------------------


class TestParseContextSwitchExtended:
    def test_returns_none_for_empty_input(self):
        assert parse_context_switch("", "duckdb") is None
        assert parse_context_switch(None, "duckdb") is None

    def test_returns_none_for_non_use_set(self):
        assert parse_context_switch("SELECT 1", "duckdb") is None

    def test_returns_none_for_invalid_use_expr(self):
        # USE with an expression that sqlglot can't parse as Use
        result = parse_context_switch("USE", "duckdb")
        assert result is None

    def test_set_database(self):
        result = parse_context_switch("SET DATABASE mydb", "duckdb")
        assert result is not None
        assert result["command"] == "SET"
        assert result["target"] == "database"
        assert result["database_name"] == "mydb"

    def test_set_schema_duckdb(self):
        result = parse_context_switch("SET SCHEMA main", "duckdb")
        assert result is not None
        assert result["target"] == "schema"

    def test_set_catalog(self):
        result = parse_context_switch("SET CATALOG mycat", "duckdb")
        assert result is not None
        assert result["target"] == "catalog"

    def test_set_with_equals(self):
        result = parse_context_switch("SET CATALOG = mycat", "snowflake")
        assert result is not None
        assert result["catalog_name"] == "mycat"

    def test_set_with_to(self):
        result = parse_context_switch("SET CATALOG TO mycat", "snowflake")
        assert result is not None
        assert result["catalog_name"] == "mycat"

    def test_set_empty_remainder_returns_none(self):
        result = parse_context_switch("SET CATALOG", "snowflake")
        assert result is None

    def test_use_catalog_keyword(self):
        result = parse_context_switch("USE CATALOG my_catalog", "starrocks")
        assert result is not None
        assert result["target"] == "catalog"

    def test_use_database_keyword(self):
        result = parse_context_switch("USE DATABASE mydb", "snowflake")
        assert result is not None
        assert result["target"] == "database"

    def test_use_schema_keyword_snowflake(self):
        result = parse_context_switch("USE SCHEMA myschema", "snowflake")
        assert result is not None
        assert result["target"] == "schema"
        assert result["schema_name"] == "myschema"

    def test_set_session_database(self):
        result = parse_context_switch("SET SESSION DATABASE mydb", "snowflake")
        assert result is not None
        assert result["target"] == "database"
        assert result["database_name"] == "mydb"

    def test_generic_use_fallback(self):
        # For postgres (has schema capability), generic fallback applies
        result = parse_context_switch("USE myschema", "postgres")
        assert result is not None

    def test_starrocks_use_catalog_dot_db(self):
        result = parse_context_switch("USE my_catalog.my_db", "starrocks")
        assert result is not None
        assert result["target"] == "database"
        assert result["catalog_name"] == "my_catalog"
        assert result["database_name"] == "my_db"


# ---------------------------------------------------------------------------
# parse_metadata_from_ddl - edge cases
# ---------------------------------------------------------------------------


class TestParseMetadataFromDDLExtended:
    def test_malformed_sql_returns_empty(self):
        result = parse_metadata_from_ddl("NOT VALID SQL AT ALL !!!!")
        assert result["table"]["name"] == ""

    def test_empty_string_returns_empty(self):
        result = parse_metadata_from_ddl("")
        assert result["table"]["name"] == ""

    def test_mysql_backtick_names(self):
        ddl = "CREATE TABLE `my_table` (`col1` INT, `col2` VARCHAR(255))"
        result = parse_metadata_from_ddl(ddl, "mysql")
        assert result["table"]["name"] == "my_table"
        assert result["columns"][0]["name"] == "col1"

    def test_column_comment(self):
        ddl = """CREATE TABLE t (
            id INT,
            name VARCHAR(100) COMMENT 'The name field'
        )"""
        result = parse_metadata_from_ddl(ddl, "mysql")
        assert result["table"]["name"] == "t"
        name_col = next((c for c in result["columns"] if c["name"] == "name"), None)
        assert name_col is not None
