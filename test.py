# simple_diagnostic.py
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 直接测试存储而不依赖配置
def main():
    try:
        print("=== Simple Storage Diagnostic ===")

        # 直接指定存储路径进行测试
        storage_path = "data/datus_db_duckdb"  # 根据您的实际路径调整
        if not os.path.exists(storage_path):
            print(f"Storage path does not exist: {storage_path}")
            return

        print(f"Storage path exists: {storage_path}")

        # 测试导入
        from datus.storage.metric.store import SemanticMetricsRAG
        rag = SemanticMetricsRAG(storage_path)

        # 测试语义模型存储
        print("\n=== Testing Semantic Model Storage ===")
        try:
            size = rag.get_semantic_model_size()
            print(f"Semantic model table size: {size}")
        except Exception as e:
            print(f"Error getting semantic model size: {e}")

        # 测试指标存储
            # 搜索所有指标
            print("\n=== 指标 ===")
            try:
                metrics_table = rag.search_all_metrics("")
                if metrics_table is not None and metrics_table.num_rows > 0:
                    print(f"找到 {metrics_table.num_rows} 个指标")

                    # 转换为字典列表
                    metrics_list = metrics_table.to_pylist()

                    for i, metric in enumerate(metrics_list[:5]):  # 只显示前5个
                        print(f"\n指标 {i + 1}:")
                        print(f"  ID: {metric.get('id', 'N/A')}")
                        print(f"  名称: {metric.get('name', 'N/A')}")
                        print(f"  语义模型: {metric.get('semantic_model_name', 'N/A')}")
                        print(f"  域名: {metric.get('domain', 'N/A')}")
                        print(f"  layer1: {metric.get('layer1', 'N/A')}")
                        print(f"  layer2: {metric.get('layer2', 'N/A')}")
                        print(
                            f"  描述: {metric.get('description', 'N/A')[:50]}..." if metric.get('description') and len(
                                metric.get('description')) > 50 else f"  描述: {metric.get('description', 'N/A')}")

                    if len(metrics_list) > 5:
                        print(f"\n... 还有 {len(metrics_list) - 5} 个指标")
                else:
                    print("未找到任何指标")

            except Exception as e:
                print(f"搜索指标时出错: {e}")
                import traceback
                traceback.print_exc()

            # 检查存储大小
            print("\n=== 存储统计 ===")
            try:
                semantic_size = rag.get_semantic_model_size()
                metrics_size = rag.get_metrics_size()
                print(f"语义模型数量: {semantic_size}")
                print(f"指标数量: {metrics_size}")
            except Exception as e:
                print(f"获取存储统计时出错: {e}")

    except Exception as e:
        print(f"Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
