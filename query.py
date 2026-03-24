import lancedb

# from sentence_transformers import SentenceTransformer

# 1. 加载向量模型（本地 HuggingFace）
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# model = SentenceTransformer("BAAI/bge-base-zh")

# 假设你本地数据库文件位于 ./my_lancedb
# db = lancedb.connect("/Users/kangxue/work/datus/Datus-agent/data/datus_db_starrocks")
# db = lancedb.connect("/Users/lyf/.datus/data/document/starrocks")
db = lancedb.connect("/Users/lyf/.datus/data/datus_db_superset")
db1 = lancedb.connect("/Users/lyf/.datus/data/sub_agents/superset_world_bank_s")
db2 = lancedb.connect("/Users/lyf/.datus/data/sub_agents/superset_world_bank_s_attribution")


# 查看有哪些表
print(db.table_names())

print(
    "$$$ semantic",
    db.open_table("semantic_model")
    .search()
    .select(
        [
            "id",
            "kind",
            "name",
            "fq_name",
            "semantic_model_name",
            "catalog_name",
            "database_name",
            "schema_name",
            "description",
        ]
    )
    .to_list(),
)

# 读取某个表
print("TOTAL:")
print("metadata:", db.open_table("schema_metadata").count_rows())
print("values:", db.open_table("schema_value").count_rows())
print("semantic_models:", db.open_table("semantic_model").count_rows())
print("metrics:", db.open_table("metrics").count_rows())
print("sqls:", db.open_table("reference_sql").count_rows())

print("$" * 100)

print("metadata:", db1.open_table("schema_metadata").count_rows())
print("values:", db1.open_table("schema_value").count_rows())
print("semantic_models:", db1.open_table("semantic_model").count_rows())
print("metrics:", db1.open_table("metrics").count_rows())
print("sqls:", db1.open_table("reference_sql").count_rows())

print("$" * 100)
metadata_table = db2.open_table("schema_metadata")
values_table = db2.open_table("schema_value")
semantic_table = db2.open_table("semantic_model")
metric_table = db2.open_table("metrics")
reference_table = db2.open_table("reference_sql")

print("metadata:", metadata_table.count_rows())
print("values:", values_table.count_rows())
print("semantic_models:", semantic_table.count_rows())
print("metrics:", metric_table.count_rows())
print("sqls:", reference_table.count_rows())
# 查询前几行
# print(table.to_pandas().head())
# print(table.schema)
# print(table2.to_pandas().head())
# print(table2.schema)


# 获取所有记录 - 使用 search().limit() 方法最可靠
for table in (metadata_table, values_table, semantic_table, metric_table, reference_table):
    print("$$$$", table.name, "START")

    rows = table.search().limit(100).to_list()
    print(f"Retrieved {len(rows)} rows")

    # 每一条记录逐行输出每一列
    for i, row in enumerate(rows, 1):
        print(f"$$$$$Record #{i}")
        for key, value in row.items():
            # if key in ["id", "name", "domain", "summary", "layer1", "layer2", "tags", "name", "filepath", "llm_text"]:
            if key != "vector":
                print(f"{key}: {value}")
        print("-" * 40)

# query_text = "IP活动"
# query_vector = model.encode(query_text).tolist()

# 4. 相似度检索
# results = (
#   table.search(query_vector)
#        .limit(3)       # 返回前 3 个最相关
#        .to_pandas()
# )
#
# for idx, row in results.iterrows():
#   print('------------' * 10)
#   print(row['semantic_model_name'])
#   print(row['name'])
#   print(f"description: {row['description']}")
#   print(f"constraint: {row['constraint']}")
#   print(f"sql_query: {row['sql_query']}")
#   print(f"_distance: {row['_distance']}")
