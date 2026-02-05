import lancedb

# from sentence_transformers import SentenceTransformer

# 1. 加载向量模型（本地 HuggingFace）
# model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# model = SentenceTransformer("BAAI/bge-base-zh")

# 假设你本地数据库文件位于 ./my_lancedb
# db = lancedb.connect("/Users/kangxue/work/datus/Datus-agent/data/datus_db_starrocks")
db = lancedb.connect("/Users/lyf/.datus/data/document/starrocks")
# db = lancedb.connect("/Users/kangxue/.datus_exp/data/datus_db_starrocks")
# db = lancedb.connect("/Users/kangxue/work/datus/Datus-agent/tests/data/datus_db_bird_school")
# db = lancedb.connect("/Users/kangxue/work/datus/Datus-agent/data/datus_db_bird_sqlite")
# db = lancedb.connect("/Users/kangxue/work/datus/Datus-agent/data/datus_db_california")
# db= lancedb.connect("/Users/kangxue/.datus_exp/data/datus_db_bird_school")
# db = lancedb.connect("/Users/kangxue/.datus/data/datus_db_duckdb")
# db = lancedb.connect("/Users/kangxue/.datus/data/datus_db_snowflake")

# 查看有哪些表
print(db.table_names())

# 读取某个表
# table = db.open_table("schema_metadata")
# table = db.open_table("schema_value")
# table = db.open_table("semantic_model")
# table = db.open_table("metrics")
# table = db.open_table("ext_knowledge")
table = db.open_table("document")
print(table.schema)

# 查询前几行
# print(table.to_pandas().head())
# print(table.schema)
# print(table2.to_pandas().head())
# print(table2.schema)


# 获取所有记录 - 使用 search().limit() 方法最可靠
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
