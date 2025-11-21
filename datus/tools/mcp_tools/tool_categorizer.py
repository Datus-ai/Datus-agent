# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

"""
Tool Categorizer for MCP Tools

This module provides intelligent categorization of user requests to determine
the most appropriate category of tools to use for fulfilling the request.
"""

import re
from typing import Dict, List, Tuple

from cachetools import TTLCache

from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class ToolCategorizer:
    """
    Intelligent tool categorizer that analyzes user requests and determines
    the most appropriate tool category to use.

    This class uses rule-based pattern matching, keyword analysis, and semantic
    understanding to categorize user requests into predefined categories.
    """

    def __init__(self, cache_ttl_seconds: int = 300):
        """
        Initialize the tool categorizer.

        Args:
            cache_ttl_seconds: TTL for categorization cache in seconds
        """
        # Cache for categorization results
        self.categorization_cache = TTLCache(maxsize=1000, ttl=cache_ttl_seconds)

        # Category definitions with keywords and patterns
        self.category_definitions = {
            "analysis": {
                "keywords": [
                    "analyze",
                    "analysis",
                    "relationship",
                    "foreign key",
                    "primary key",
                    "join",
                    "correlation",
                    "dependency",
                    "lineage",
                    "statistics",
                    "distribution",
                    "pattern",
                    "trend",
                    "insight",
                    "explore relationship",
                    "table relationship",
                    # Chinese keywords
                    "分析",
                    "关系",
                    "统计",
                    "趋势",
                    "模式",
                    "依赖",
                    "关联",
                    "血缘",
                ],
                "patterns": [
                    r".*关系.*分析.*",
                    r".*ER.*关系.*",
                    r".*table.*relationship.*",
                    r".*foreign.*key.*",
                    r".*primary.*key.*",
                    r".*analyze.*relationship.*",
                    r".*discover.*relationship.*",
                ],
                "description": "Deep analysis of data structures, relationships, and patterns",
            },
            "exploration": {
                "keywords": [
                    "list",
                    "show",
                    "describe",
                    "get",
                    "fetch",
                    "retrieve",
                    "sample",
                    "browse",
                    "explore",
                    "view",
                    "display",
                    "schema",
                    "table",
                    "column",
                    "metadata",
                    "structure",
                    "info",
                    "information",
                    # Chinese keywords
                    "列出",
                    "显示",
                    "获取",
                    "查看",
                    "浏览",
                    "表",
                    "列",
                    "模式",
                    "结构",
                    "信息",
                ],
                "patterns": [
                    r".*list.*tables?.*",
                    r".*show.*schema.*",
                    r".*describe.*table.*",
                    r".*get.*info.*",
                    r".*what.*tables?.*",
                    r".*browse.*data.*",
                ],
                "description": "Basic exploration and browsing of data structures",
            },
            "management": {
                "keywords": [
                    "switch",
                    "change",
                    "manage",
                    "admin",
                    "administration",
                    "configure",
                    "setup",
                    "instance",
                    "job",
                    "history",
                    "status",
                    "system",
                    "control",
                    "operation",
                    # Chinese keywords
                    "切换",
                    "管理",
                    "配置",
                    "实例",
                    "作业",
                    "历史",
                    "状态",
                    "系统",
                ],
                "patterns": [
                    r".*switch.*instance.*",
                    r".*change.*instance.*",
                    r".*job.*history.*",
                    r".*system.*status.*",
                    # monitoring-related patterns live under the 'monitoring' category
                ],
                "description": "System management, configuration, and operational tasks",
            },
            "generation": {
                "keywords": [
                    "generate",
                    "create",
                    "build",
                    "make",
                    "produce",
                    "construct",
                    "develop",
                    "design",
                    "SQL",
                    "query",
                    "model",
                    "dashboard",
                    "report",
                    "semantic model",
                    "data model",
                    # Chinese keywords
                    "生成",
                    "创建",
                    "构建",
                    "制作",
                    "开发",
                    "设计",
                    "查询",
                    "模型",
                    "报表",
                ],
                "patterns": [
                    r".*generate.*SQL.*",
                    r".*create.*model.*",
                    r".*build.*dashboard.*",
                    r".*make.*query.*",
                    r".*produce.*report.*",
                ],
                "description": "Generation of SQL, models, reports, and other artifacts",
            },
            "monitoring": {
                "keywords": [
                    "monitor",
                    "track",
                    "watch",
                    "observe",
                    "check",
                    "verify",
                    "validate",
                    "test",
                    "performance",
                    "metrics",
                    "health",
                    "alert",
                    "notification",
                    "log",
                    "audit",
                    # Chinese keywords
                    "监控",
                    "跟踪",
                    "检查",
                    "验证",
                    "测试",
                    "性能",
                    "指标",
                    "健康",
                    "日志",
                ],
                "patterns": [
                    r".*monitor.*performance.*",
                    r".*check.*metrics.*",
                    r".*track.*health.*",
                    r".*watch.*system.*",
                    r".*observe.*behavior.*",
                ],
                "description": "Monitoring, tracking, and health checking activities",
            },
        }

        # Default fallback category
        self.default_category = "exploration"

        logger.info("Initialized Tool Categorizer with categories: " + ", ".join(self.category_definitions.keys()))

    def categorize_user_request(self, user_message: str) -> str:
        """
        Categorize a user request into the most appropriate tool category.

        Args:
            user_message: The user's request message

        Returns:
            The determined category name
        """
        if not user_message or not user_message.strip():
            return self.default_category

        # Check cache first
        message_hash = hash(user_message.lower().strip())
        if message_hash in self.categorization_cache:
            logger.debug(f"Using cached categorization for message hash: {message_hash}")
            return self.categorization_cache[message_hash]

        try:
            # Analyze the message and determine category
            category = self._analyze_message(user_message)

            # Cache the result
            self.categorization_cache[message_hash] = category

            logger.debug(f"Categorized message as '{category}': {user_message[:100]}...")
            return category

        except Exception as e:
            logger.error(f"Error categorizing user request: {e}")
            return self.default_category

    def _analyze_message(self, user_message: str) -> str:
        """
        Analyze the user message to determine the best category.

        Args:
            user_message: The user's message to analyze

        Returns:
            The determined category
        """
        message_lower = user_message.lower().strip()
        category_scores = {}

        # Score each category based on keyword and pattern matching
        for category, definition in self.category_definitions.items():
            score = 0

            # Keyword matching
            keyword_matches = self._count_keyword_matches(message_lower, definition["keywords"])
            score += keyword_matches * 2  # Keywords get double weight

            # Pattern matching
            pattern_matches = self._count_pattern_matches(message_lower, definition["patterns"])
            score += pattern_matches * 3  # Patterns get triple weight

            category_scores[category] = score

        # Special handling for relationship analysis
        if self._is_relationship_analysis_request(message_lower):
            category_scores["analysis"] = category_scores.get("analysis", 0) + 10

        # Find the category with the highest score
        if category_scores:
            best_category = max(category_scores, key=category_scores.get)
            best_score = category_scores[best_category]

            # Only return the best category if it has a meaningful score
            if best_score > 0:
                logger.debug(f"Category scores: {category_scores}")
                return best_category

        # Return default category if no clear match
        logger.debug(f"No clear category match, using default: {self.default_category}")
        return self.default_category

    def _count_keyword_matches(self, message: str, keywords: List[str]) -> int:
        """
        Count the number of keyword matches in the message.

        Args:
            message: The message to search in
            keywords: List of keywords to search for

        Returns:
            Number of keyword matches found
        """
        matches = 0
        for keyword in keywords:
            if keyword.lower() in message:
                matches += 1
                # Bonus for exact word matches
                if re.search(r"\b" + re.escape(keyword.lower()) + r"\b", message):
                    matches += 1

        return matches

    def _count_pattern_matches(self, message: str, patterns: List[str]) -> int:
        """
        Count the number of pattern matches in the message.

        Args:
            message: The message to search in
            patterns: List of regex patterns to search for

        Returns:
            Number of pattern matches found
        """
        matches = 0
        for pattern in patterns:
            try:
                if re.search(pattern, message, re.IGNORECASE):
                    matches += 1
            except re.error:
                logger.warning(f"Invalid regex pattern: {pattern}")
                continue

        return matches

    def _is_relationship_analysis_request(self, message: str) -> bool:
        """
        Special detection for relationship analysis requests.

        Args:
            message: The message to analyze

        Returns:
            True if this appears to be a relationship analysis request
        """
        # Specific patterns that strongly indicate relationship analysis
        relationship_indicators = [
            r".*表.*之间.*关系.*",  # Chinese: tables between relationships
            r".*ER.*关系.*",  # ER relationships
            r".*table.*relationship.*",
            r".*foreign.*key.*relationship.*",
            r".*discover.*relationship.*",
            r".*analyze.*relationship.*",
            r".*schema.*relationship.*",
        ]

        for pattern in relationship_indicators:
            try:
                if re.search(pattern, message, re.IGNORECASE):
                    return True
            except re.error:
                continue

        return False

    def get_category_confidence(self, user_message: str) -> Tuple[str, float]:
        """
        Get the categorization result with confidence score.

        Args:
            user_message: The user's request message

        Returns:
            Tuple of (category, confidence_score) where confidence is 0.0-1.0
        """
        if not user_message or not user_message.strip():
            return self.default_category, 0.0

        message_lower = user_message.lower().strip()
        category_scores = {}
        total_possible_score = 0

        # Calculate scores and maximum possible score
        for category, definition in self.category_definitions.items():
            score = 0
            max_score = 0

            # Keyword scoring
            keyword_matches = self._count_keyword_matches(message_lower, definition["keywords"])
            score += keyword_matches * 2
            max_score += len(definition["keywords"]) * 4  # Max possible keyword score

            # Pattern scoring
            pattern_matches = self._count_pattern_matches(message_lower, definition["patterns"])
            score += pattern_matches * 3
            max_score += len(definition["patterns"]) * 3  # Max possible pattern score

            category_scores[category] = score
            total_possible_score = max(total_possible_score, max_score)

        # Special handling for relationship analysis
        if self._is_relationship_analysis_request(message_lower):
            category_scores["analysis"] = category_scores.get("analysis", 0) + 10
            total_possible_score += 10

        # Find best category and calculate confidence
        if category_scores and total_possible_score > 0:
            best_category = max(category_scores, key=category_scores.get)
            best_score = category_scores[best_category]
            confidence = min(best_score / total_possible_score, 1.0)

            return best_category, confidence

        return self.default_category, 0.0

    def add_category_keywords(self, category: str, keywords: List[str]) -> None:
        """
        Add new keywords to an existing category.

        Args:
            category: The category to add keywords to
            keywords: List of keywords to add
        """
        if category in self.category_definitions:
            existing_keywords = set(self.category_definitions[category]["keywords"])
            existing_keywords.update(keywords)
            self.category_definitions[category]["keywords"] = list(existing_keywords)

            logger.info(f"Added {len(keywords)} keywords to category '{category}'")
        else:
            logger.warning(f"Category '{category}' does not exist")

    def add_category_patterns(self, category: str, patterns: List[str]) -> None:
        """
        Add new regex patterns to an existing category.

        Args:
            category: The category to add patterns to
            patterns: List of regex patterns to add
        """
        if category in self.category_definitions:
            existing_patterns = set(self.category_definitions[category]["patterns"])
            existing_patterns.update(patterns)
            self.category_definitions[category]["patterns"] = list(existing_patterns)

            logger.info(f"Added {len(patterns)} patterns to category '{category}'")
        else:
            logger.warning(f"Category '{category}' does not exist")

    def get_category_definitions(self) -> Dict:
        """
        Get the current category definitions.

        Returns:
            Dictionary of category definitions
        """
        return self.category_definitions.copy()

    def clear_cache(self) -> None:
        """Clear the categorization cache."""
        self.categorization_cache.clear()
        logger.info("Cleared categorization cache")
