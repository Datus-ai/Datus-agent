from datus.utils.constants import DBType, SQLType
from datus.utils.json_utils import llm_result2json
from datus.utils.sql_utils import extract_table_names, parse_metadata, parse_sql_type, parse_table_name_parts

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
    table_meta = parse_metadata(SQL, DBType.SNOWFLAKE)
    print(table_meta)
    assert table_meta["table"]["name"] == "VARIANTS"
    assert table_meta["columns"][0]["name"] == "reference_name"
    assert len(table_meta["columns"]) == 40


def test_parse_oracle():
    table_meta = parse_metadata(
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
    table_meta = parse_metadata(
        """CREATE TABLE `connected` (
`atom_id` varchar(256) NOT NULL,
`atom_id2` varchar(256) NOT NULL,
`bond_id` varchar(256) NULL,
  PRIMARY KEY (`atom_id`, `atom_id2`),
  FOREIGN KEY (`atom_id`) REFERENCES `atom`(`atom_id`),
  FOREIGN KEY (`atom_id2`) REFERENCES `atom`(`atom_id`),
  FOREIGN KEY (`bond_id`) REFERENCES `bond`(`bond_id`)
);""",
        DBType.MYSQL,
    )
    assert table_meta["table"]["name"] == "connected"
    assert table_meta["columns"][0]["name"] == "atom_id"
    assert len(table_meta["columns"]) == 3


def test_parse_postgresql():
    table_meta = parse_metadata(
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
        DBType.POSTGRES,
    )
    assert table_meta["table"]["name"] == "trans"
    assert table_meta["columns"][0]["name"] == "trans_id"
    assert len(table_meta["columns"]) == 10


def test_parse_sqlserver():
    table_meta = parse_metadata(
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
        DBType.SQLSERVER,
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
    tables = extract_table_names(select_sql, dialect)
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
        dialect=DBType.SNOWFLAKE,
    )

    print("-" * 100)
    parse_and_print(
        """SELECT account_id, MAX(payments) AS max_payment, MIN(payments) AS min_payment
        FROM loan GROUP BY account_id HAVING COUNT(account_id) > 1 AND (MAX(payments) - MIN(payments)) > 2;
        WITH cte AS (SELECT * FROM loan)
        SELECT * FROM cte;
        """,
        ["loan"],
        dialect=DBType.POSTGRES,
    )


def test_parse_duckdb():
    table_meta = parse_metadata(
        """CREATE TABLE abc.test (
id bigint primary key,
account_id bigint null default '0',
date date null,
type text null)""",
        dialect=DBType.DUCKDB,
    )
    print(table_meta)


def test_parse_sqlite():
    table_meta = parse_metadata(
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


def test_parse_full_tables():
    table_meta = parse_table_name_parts("test.abc", dialect=DBType.DUCKDB)
    assert table_meta["schema_name"] == "test"
    assert table_meta["table_name"] == "abc"
    assert table_meta["database_name"] == ""
    assert table_meta["catalog_name"] == ""

    table_meta = parse_table_name_parts("`test`.abc", dialect=DBType.MYSQL)
    assert table_meta["schema_name"] == "test"
    assert table_meta["table_name"] == "abc"
    assert table_meta["database_name"] == ""
    assert table_meta["catalog_name"] == ""

    table_meta = parse_table_name_parts('''TEST_DB."test_schema"."abc"''', dialect=DBType.SNOWFLAKE)
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
    assert parse_sql_type(merge_sql, dialect=DBType.SNOWFLAKE) == SQLType.MERGE

    assert parse_sql_type("EXPLAIN SELECT * FROM gold_vs_bitcoin", dialect=DBType.DUCKDB) == SQLType.EXPLAIN

    assert parse_sql_type("SHOW TABLES", dialect=DBType.DUCKDB) == SQLType.METADATA_SHOW

    assert parse_sql_type("SHOW CATALOGS", dialect="starrocks") == SQLType.METADATA_SHOW
