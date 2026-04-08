# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# Re-exports from datus_bi_core for backward compatibility
try:
    from datus_bi_core import (  # noqa: F401
        AuthParam,
        AuthType,
        BIAdapterBase,
        BIAdapterRegistry,
        ChartInfo,
        ChartWriteMixin,
        ColumnInfo,
        DashboardInfo,
        DashboardWriteMixin,
        DatasetInfo,
        DatasetWriteMixin,
        DimensionDef,
        MetricDef,
        QuerySpec,
        adapter_registry,
    )
except ImportError:
    pass  # datus-bi-core is an optional dependency
