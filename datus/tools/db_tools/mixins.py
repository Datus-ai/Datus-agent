# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# See http://www.apache.org/licenses/LICENSE-2.0 for details.

from abc import ABC, abstractmethod
from typing import Dict, List


class CatalogSupportMixin(ABC):
    """
    Mixin for databases that support catalog namespace.

    Databases with catalog support: Snowflake, StarRocks, etc.
    """

    @abstractmethod
    def get_catalogs(self) -> List[str]:
        """Get list of available catalogs.

        Returns:
            List of catalog names
        """
        raise NotImplementedError

    @abstractmethod
    def switch_catalog(self, catalog_name: str) -> None:
        """Switch to a different catalog.

        Args:
            catalog_name: Name of the catalog to switch to
        """
        raise NotImplementedError

    @abstractmethod
    def default_catalog(self) -> str:
        """Get the default catalog name.

        Returns:
            Default catalog name
        """
        raise NotImplementedError


class MaterializedViewSupportMixin(ABC):
    """
    Mixin for databases that support materialized views.

    Databases with materialized view support: Snowflake, PostgreSQL, StarRocks, etc.
    """

    @abstractmethod
    def get_materialized_views(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = ""
    ) -> List[str]:
        """Get list of materialized view names.

        Args:
            catalog_name: Optional catalog name
            database_name: Optional database name
            schema_name: Optional schema name

        Returns:
            List of materialized view names
        """
        raise NotImplementedError

    @abstractmethod
    def get_materialized_views_with_ddl(
        self, catalog_name: str = "", database_name: str = "", schema_name: str = ""
    ) -> List[Dict[str, str]]:
        """Get materialized views with their DDL definitions.

        Args:
            catalog_name: Optional catalog name
            database_name: Optional database name
            schema_name: Optional schema name

        Returns:
            List of dictionaries containing materialized view metadata and DDL
        """
        raise NotImplementedError


class SchemaNamespaceMixin(ABC):
    """
    Mixin for databases that support schema-level namespace.

    Databases with schema namespace: PostgreSQL, Snowflake, etc.
    Note: MySQL does not have schema namespace (database == schema).
    """

    @abstractmethod
    def get_schemas(self, catalog_name: str = "", database_name: str = "", include_sys: bool = False) -> List[str]:
        """Get list of schema names.

        Args:
            catalog_name: Optional catalog name
            database_name: Optional database name
            include_sys: Whether to include system schemas

        Returns:
            List of schema names
        """
        raise NotImplementedError
