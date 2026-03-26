我之前根据 [agent_for_saas](.claude/agent_for_saas.md) 做了一些改造，但针对storage这里，我需要做一些调整:

1. 回滚 storage 中扩展字段这里的逻辑，统一增加字段 datasource_id , creator_id, updator_id 字段。其中datasource_id 如果没有赋值，取
   namespace； creator_id 和 updator_id 默认值为 datus_agent;
2. 将所有 storage 改回单例，查询时 增加 datasource_id 字段；RAG层负责传递 datasource_id字段
3. 其它地方 尽量使用 RAG，而不是Storage；
4. 去掉现在 lancedb 中 datus_db_{namespace} 的逻辑，统一路径为 datus_db，通过 datasource_id字段物理隔离
5. 提供迁移脚本，将现在 datus_db_{namespace} 中的数据，迁移到 datus_db 中