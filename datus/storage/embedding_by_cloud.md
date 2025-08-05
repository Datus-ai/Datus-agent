# Description of each LLM

## OpenAI
### Window Context
1. gpt4.1: 1,047,576
2. 4o/4o-mini: 200,000
### Embeddings
| Model Name             | dimensions size |
|------------------------|:---------------:|
| text-embedding-3-small |      1536       |
| text-embedding-3-large |      3072       |


## DeepSeek

### Window Context
64k
### Embeddings
Unsupported

## Claude

### Window Context
200K

### Embeddings
| Model Name       |  dimensions size   |                                                                                                                  Description                                                                                                                  |
|------------------|:------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| voyage-3-large   | 256/512/1024/2048  |                                                The best general-purpose and multilingual retrieval quality. See [blog post](https://blog.voyageai.com/2025/01/07/voyage-3-large/) for details.                                                |
| voyage-3.5       | 256/512/1024/2048  |                                               Optimized for general-purpose and multilingual retrieval quality. See [blog post](https://blog.voyageai.com/2025/05/20/voyage-3-5/) for details.                                                |
| voyage-3.5-lite  | 256/512/1024/2048  |                                                                Optimized for latency and cost. See [blog post](https://blog.voyageai.com/2025/05/20/voyage-3-5/) for details.                                                                 |
| voyage-3         | 256/512/1024/2048  |                                                                Optimized for code retrieval. See [blog post](https://blog.voyageai.com/2024/12/04/voyage-code-3/) for details.                                                                |
| voyage-finance-2 | 256/512/1024/2048  |                                   Optimized for finance retrieval and RAG. See [blog post](https://blog.voyageai.com/2024/06/03/domain-specific-embeddings-finance-edition-voyage-finance-2/) for details.                                    |
| voyage-law-2     | 256/512/1024/2048  | Optimized for legal and long-context retrieval and RAG. Also improved performance across all domains. See [blog post](https://blog.voyageai.com/2024/04/15/domain-specific-embeddings-and-retrieval-legal-edition-voyage-law-2/) for details. |


## Google

### Window Context
1. gemini 2.0 flash: input 1,048,576; output 8192
2. gemini 2.5 pro: input	1,048,576; output 64,000
3. gemini 2.0 flash thinking: input	1,048,576; output 65536

### Embeddings
| Model Name                       |  dimensions size  |                                                                                                                                                                  Description                                                                                                                                                                   |
|----------------------------------|:-----------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| gemini-embedding-001             |       2048        | State-of-the-art performance across English, multilingual and code tasks. It unifies the previously specialized models like text-embedding-005 and text-multilingual-embedding-002 and achieves better performance in their respective domains. Read our [Tech Report](https://deepmind.google/research/publications/157741/) for more detail. |
| text-embedding-005               |       2048        |                                                                                                                                                     Specialized in English and code tasks.                                                                                                                                                     |
| text-multilingual-embedding-002  |       2048        |                                                                                                                                                      Specialized in multilingual tasks.                                                                                                                                                        |


## Kimi
### Window Context
kimi-k2: 128k
### Embeddings
Unsupported

## Qwen(Aliyun)

### Window Context
1. qwen-plus: 128K
2. qwen-turbo: 1M
3. qwen3(4b, 8b, 14b, 32b, 30b-a3b, 235b-a22b): 128k
4. qwen3-coder: 1M
5. qwen-max: 32k
6. qwen-max-latest: 128k

### Embeddings

| Model Name                         | dimensions size | Description  |
|------------------------------------|:---------------:|:------------:|
| text-embedding-v4(qwen3-embedding) | 1024/1536/2048  | multilingual |
| text-embedding-v3                  |    768/ 1024    | multilingual |
| text-embedding-v2                  |      1536       | multilingual |
| text-embedding-v1                  |      1536       | multilingual |

