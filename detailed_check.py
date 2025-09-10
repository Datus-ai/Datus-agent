import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datus.configuration.agent_config import AgentConfig
from datus.storage.metric.store import rag_by_configuration


def main():
    try:
        # 加载配置
        config = AgentConfig.load_from_file("conf/agent.yml")
        rag = rag_by_configuration(config)

        print(f"Storage path: {rag.db_path}")
        print(f"Storage exists: {os.path.exists(rag.db_path)}")

        if os.path.exists(rag.db_path):
            print("Storage contents:")
            for item in os.listdir(rag.db_path):
                print(f"  {item}")

        # 检查语义模型存储
        print("\n=== Checking Semantic Model Storage ===")
        try:
            semantic_size = rag.get_semantic_model_size()
            print(f"Semantic model table size: {semantic_size}")
        except Exception as e:
            print(f"Error getting semantic model size: {e}")

        # 检查指标存储
        print("\n=== Checking Metrics Storage ===")
        try:
            metrics_size = rag.get_metrics_size()
            print(f"Metrics table size: {metrics_size}")
        except Exception as e:
            print(f"Error getting metrics size: {e}")

        # 尝试直接访问表
        print("\n=== Direct Table Access ===")
        try:
            # 检查语义模型表
            rag.semantic_model_storage._ensure_table_ready()
            if rag.semantic_model_storage.table is not None:
                print("Semantic model table is accessible")
                print(f"Semantic model table count: {rag.semantic_model_storage.table.count_rows()}")
                # 显示前几行
                if rag.semantic_model_storage.table.count_rows() > 0:
                    sample_data = rag.semantic_model_storage.table.head(3).to_pylist()
                    print("Sample semantic model data:")
                    for i, row in enumerate(sample_data):
                        print(f"  Row {i + 1}: {row}")
            else:
                print("Semantic model table is None")
        except Exception as e:
            print(f"Error accessing semantic model table: {e}")
            import traceback
            traceback.print_exc()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
