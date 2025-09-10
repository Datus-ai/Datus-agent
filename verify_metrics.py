# read_semantic_models_and_metrics.py
"""
读取semantic_model和metrics的示例脚本
该脚本演示了如何从存储中读取语义模型和指标数据
"""

import json
from typing import Dict, List, Any
from pathlib import Path

# 假设我们使用与MetricsScreen相同的存储类
try:
    from datus.storage.metric.store import SemanticModelStorage, MetricStorage
    from datus.storage.config import StorageConfig
except ImportError:
    # 如果无法导入实际的存储类，我们创建模拟类用于演示
    class SemanticModelStorage:
        def search_all(self, query: str) -> List[Dict]:
            # 模拟返回一些示例数据
            return [
                {
                    "semantic_model_name": "transactions",
                    "semantic_model_desc": "Transaction data model",
                    "domain": "finance",
                    "catalog_name": "main_catalog",
                    "database_name": "finance_db",
                    "schema_name": "public",
                    "table_name": "transactions",
                    "layer1": "gold",
                    "layer2": "duck",
                    "catalog_database_schema": "gold_duck",
                    "identifiers": json.dumps([
                        {"name": "transaction_id", "type": "primary"}
                    ]),
                    "dimensions": json.dumps([
                        {"name": "date", "type": "date"},
                        {"name": "category", "type": "string"}
                    ]),
                    "measures": json.dumps([
                        {"name": "amount", "agg": "sum"},
                        {"name": "count", "agg": "count"}
                    ])
                },
                {
                    "semantic_model_name": "users",
                    "semantic_model_desc": "User information model",
                    "domain": "user_analytics",
                    "catalog_name": "main_catalog",
                    "database_name": "analytics_db",
                    "schema_name": "public",
                    "table_name": "users",
                    "layer1": "silver",
                    "layer2": "customer",
                    "catalog_database_schema": "silver_customer",
                    "identifiers": json.dumps([
                        {"name": "user_id", "type": "primary"}
                    ]),
                    "dimensions": json.dumps([
                        {"name": "signup_date", "type": "date"},
                        {"name": "country", "type": "string"}
                    ]),
                    "measures": json.dumps([
                        {"name": "user_count", "agg": "count_distinct"}
                    ])
                }
            ]


    class MetricStorage:
        def search_all(self, semantic_model_name: str, select_fields: Any = None):
            # 模拟返回指标数据
            if semantic_model_name == "transactions":
                # 模拟PyArrow Table结构
                class MockTable:
                    def __init__(self):
                        self.num_rows = 2
                        self.schema = type('Schema', (),
                                           {'names': ['name', 'description', 'sql_query', 'constraint']})()

                    def __getitem__(self, key):
                        data = {
                            'name': ['total_amount', 'transaction_count'],
                            'description': ['Total transaction amount', 'Number of transactions'],
                            'sql_query': ['SELECT SUM(amount) FROM transactions', 'SELECT COUNT(*) FROM transactions'],
                            'constraint': ['sum', 'count']
                        }
                        return [type('Value', (), {'as_py': lambda: v})() for v in data[key]]

                return MockTable()
            return None


def read_semantic_models_and_metrics():
    """
    读取所有语义模型和相关指标
    """
    print("开始读取语义模型和指标数据...")

    # 初始化存储
    semantic_model_storage = SemanticModelStorage()
    metric_storage = MetricStorage()

    # 读取所有语义模型
    print("\n1. 读取语义模型...")
    semantic_models = semantic_model_storage.search_all("")

    if not semantic_models:
        print("未找到任何语义模型")
        return

    print(f"找到 {len(semantic_models)} 个语义模型:")

    # 存储所有数据的字典
    all_data = {}

    # 遍历每个语义模型
    for i, model in enumerate(semantic_models, 1):
        model_name = model.get('semantic_model_name', 'Unknown')
        print(f"\n  {i}. 语义模型: {model_name}")

        # 显示语义模型基本信息
        print(f"     描述: {model.get('semantic_model_desc', 'N/A')}")
        print(f"     域: {model.get('domain', 'N/A')}")
        print(
            f"     数据库表: {model.get('catalog_name', 'N/A')}.{model.get('database_name', 'N/A')}.{model.get('schema_name', 'N/A')}.{model.get('table_name', 'N/A')}")
        print(f"     分层: {model.get('layer1', 'N/A')}/{model.get('layer2', 'N/A')}")

        # 解析并显示结构化信息
        try:
            identifiers = json.loads(model.get('identifiers', '[]'))
            dimensions = json.loads(model.get('dimensions', '[]'))
            measures = json.loads(model.get('measures', '[]'))

            print(f"     标识符数量: {len(identifiers)}")
            print(f"     维度数量: {len(dimensions)}")
            print(f"     度量数量: {len(measures)}")
        except json.JSONDecodeError:
            print("     无法解析结构化信息")

        # 为该语义模型读取指标
        print(f"     读取指标...")
        metrics_data = []

        try:
            metrics_table = metric_storage.search_all(model_name, select_fields=None)

            if metrics_table and metrics_table.num_rows > 0:
                # 转换PyArrow Table为字典列表
                field_names = metrics_table.schema.names
                for j in range(metrics_table.num_rows):
                    row_dict = {}
                    for field_name in field_names:
                        value = metrics_table[field_name][j]
                        # 处理PyArrow值
                        if hasattr(value, 'as_py'):
                            row_dict[field_name] = value.as_py()
                        else:
                            row_dict[field_name] = value
                    metrics_data.append(row_dict)

                print(f"     找到 {len(metrics_data)} 个指标:")
                for metric in metrics_data:
                    print(f"       - {metric.get('name', 'Unknown')}: {metric.get('description', 'N/A')}")
            else:
                print(f"     未找到指标")

        except Exception as e:
            print(f"     读取指标时出错: {str(e)}")

        # 将数据存储到字典中
        all_data[model_name] = {
            'semantic_model': model,
            'metrics': metrics_data
        }

    # 保存数据到文件
    output_file = "semantic_models_and_metrics.json"
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        print(f"\n数据已保存到 {output_file}")
    except Exception as e:
        print(f"保存数据时出错: {str(e)}")

    return all_data


def display_hierarchical_structure(data: Dict):
    """
    按分层结构显示数据
    """
    print("\n" + "=" * 50)
    print("分层结构视图")
    print("=" * 50)

    # 按domain分组
    domain_groups = {}
    for model_name, model_data in data.items():
        semantic_model = model_data['semantic_model']
        domain = semantic_model.get('domain', 'unknown_domain')

        if domain not in domain_groups:
            domain_groups[domain] = {}

        layer1 = semantic_model.get('layer1', 'default_layer1')
        if layer1 not in domain_groups[domain]:
            domain_groups[domain][layer1] = {}

        layer2 = semantic_model.get('layer2', 'default_layer2')
        if layer2 not in domain_groups[domain][layer1]:
            domain_groups[domain][layer1][layer2] = []

        domain_groups[domain][layer1][layer2].append(model_data)

    # 显示分层结构
    for domain, layer1_groups in domain_groups.items():
        print(f"\n📁 Domain: {domain}")
        for layer1, layer2_groups in layer1_groups.items():
            print(f"  📂 Layer1: {layer1}")
            for layer2, models in layer2_groups.items():
                print(f"    📂 Layer2: {layer2}")
                for model_data in models:
                    semantic_model = model_data['semantic_model']
                    model_name = semantic_model.get('semantic_model_name', 'Unknown')
                    print(f"      📊 {model_name}")
                    metrics = model_data['metrics']
                    if metrics:
                        for metric in metrics:
                            print(f"        • {metric.get('name', 'Unknown Metric')}")


if __name__ == "__main__":
    # 读取数据
    data = read_semantic_models_and_metrics()

    if data:
        # 显示分层结构
        display_hierarchical_structure(data)

        print(f"\n总共处理了 {len(data)} 个语义模型")
    else:
        print("未读取到任何数据")
