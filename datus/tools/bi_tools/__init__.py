# Copyright 2025-present DatusAI, Inc.
# Licensed under the Apache License, Version 2.0.
# Backward-compatible shim: re-exports from datus_bi_core
try:
    from datus_bi_core import (  # noqa: F401
        AuthParam,
        AuthType,
        BIAdaptorBase,
        BIAdaptorRegistry,
        ChartInfo,
        ChartWriteMixin,
        ColumnInfo,
        DashboardInfo,
        DashboardWriteMixin,
        DatasetInfo,
        DatasetWriteMixin,
        DimensionDef,
        ListDashboardsMixin,
        MetricDef,
        QuerySpec,
        adaptor_registry,
    )
except ImportError:
    # datus_bi_core not installed yet; fall back to local definitions
    from datus.tools.bi_tools.base_adaptor import (  # noqa: F401
        AuthParam,
        AuthType,
        BIAdaptorBase,
        ChartInfo,
        ColumnInfo,
        DashboardInfo,
        DatasetInfo,
        DimensionDef,
        MetricDef,
        QuerySpec,
    )
    from datus.tools.bi_tools.registry import BIAdaptorRegistry, adaptor_registry  # noqa: F401

    ListDashboardsMixin = None
    DashboardWriteMixin = None
    ChartWriteMixin = None
    DatasetWriteMixin = None
