from typing import Set

from datus.storage.ext_knowledge.store import ExtKnowledgeStore


def exists_ext_knowledge(storage: ExtKnowledgeStore, build_mode: str = "overwrite") -> Set[str]:
    """Get existing external knowledge IDs based on build mode.

    Args:
        storage: ExtKnowledgeStore instance
        build_mode: "overwrite" to ignore existing data, "incremental" to check existing

    Returns:
        Set of existing knowledge IDs
    """
    existing_knowledge = set()
    if build_mode == "overwrite":
        return existing_knowledge

    if build_mode == "incremental":
        # Get all existing knowledge entries to avoid duplicates
        all_knowledge = storage.search_all_knowledge()
        for knowledge in all_knowledge:
            knowledge_id = gen_ext_knowledge_id(
                knowledge["domain"], knowledge["layer1"], knowledge["layer2"], knowledge["terminology"]
            )
            existing_knowledge.add(knowledge_id)

    return existing_knowledge


def gen_ext_knowledge_id(domain: str, layer1: str, layer2: str, terminology: str) -> str:
    """Generate unique ID for external knowledge entry.

    Args:
        domain: Business domain
        layer1: First layer categorization
        layer2: Second layer categorization
        terminology: Business terminology/concept

    Returns:
        Unique knowledge ID
    """
    # Clean inputs to avoid issues with special characters
    clean_domain = domain.replace(" ", "_").replace("/", "_")
    clean_layer1 = layer1.replace(" ", "_").replace("/", "_")
    clean_layer2 = layer2.replace(" ", "_").replace("/", "_")
    clean_terminology = terminology.replace(" ", "_").replace("/", "_")

    return f"{clean_domain}_{clean_layer1}_{clean_layer2}_{clean_terminology}"
