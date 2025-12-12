import json
import time
from typing import Any, Dict, Generator, List, Optional, Union
from urllib.parse import parse_qs, urlparse

import httpx

from datus.tools.bi_tools.base_adaptor import AuthType, BiAdaptorBase, QuestionSqlPair
from datus.utils.loggings import get_logger

logger = get_logger(__name__)


class SupersetAdaptorError(RuntimeError):
    """Errors raised by the Superset adaptor."""


class SupersetAdaptor(BiAdaptorBase):
    """Adaptor that extracts chart SQL from a Superset dashboard."""

    def __init__(
        self,
        base_url: str,
        auth_params: Union[str, Dict[str, Any]],
        auth_type: AuthType = AuthType.LOGIN,
        *,
        verify_ssl: bool = True,
        timeout: Union[float, httpx.Timeout] = 30.0,
        session: Optional[httpx.Client] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_base = self.base_url
        if not self._api_base.endswith("/api/v1"):
            self._api_base = f"{self._api_base}/api/v1"

        self.auth_type = auth_type
        self.auth_params = auth_params

        self._client = session or httpx.Client(timeout=timeout, verify=verify_ssl, follow_redirects=True)
        self._owns_client = session is None

        self._auth_header_value: Optional[str] = None
        self._token_expiration: Optional[float] = None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def parse_sql_pair(self, dashboard_url: str) -> Generator[QuestionSqlPair, None, None]:
        dashboard_id = self._extract_dashboard_id(dashboard_url)
        dashboard_meta = self._get_dashboard(dashboard_id)
        charts = self._get_dashboard_charts(dashboard_id)

        for chart_meta in charts:
            form_data = chart_meta.get("form_data", {})
            chart_id = (
                form_data.get("slice_id")
                or chart_meta.get("slice_id")
                or chart_meta.get("chart_id")
                or chart_meta.get("id")
            )
            if chart_id is None:
                logger.warning(f"Skip chart without chat_id: {chart_meta}")
                continue

            try:
                chart_detail = self._get_chart(chart_id)
                query_context = self._extract_query_context(chart_detail)
                sql_candidates = self._collect_sql_from_chart(chart_id, query_context)
            except SupersetAdaptorError as exc:
                logger.warning(f"Failed to fetch SQL for chart {chart_id}: {exc}")
                yield QuestionSqlPair(
                    chart_id=str(chart_id),
                    title=self._chart_title(chart_meta, chart_detail=None),
                    description=self._chart_description(chart_meta, chart_detail=None),
                    sql="",
                    origin="error",
                    extra={
                        "dashboard_id": dashboard_id,
                        "chart_meta": chart_meta,
                        "error": str(exc),
                    },
                )
                continue

            if not sql_candidates:
                yield QuestionSqlPair(
                    chart_id=str(chart_id),
                    title=self._chart_title(chart_meta, chart_detail=chart_detail),
                    description=self._chart_description(chart_meta, chart_detail=chart_detail),
                    sql="",
                    origin="missing",
                    extra={
                        "dashboard_id": dashboard_id,
                        "chart_meta": chart_meta,
                        "chart_detail": chart_detail,
                        "reason": "Superset API did not return SQL for this chart",
                    },
                )
                continue

            for idx, sql_text in enumerate(sql_candidates):
                pair_id = str(chart_id) if idx == 0 else f"{chart_id}:{idx + 1}"
                yield QuestionSqlPair(
                    chart_id=pair_id,
                    title=self._chart_title(chart_meta, chart_detail=chart_detail, query_index=idx),
                    description=self._chart_description(chart_meta, chart_detail=chart_detail),
                    sql=sql_text,
                    origin="native",
                    extra={
                        "dashboard_id": dashboard_id,
                        "dashboard": {
                            "id": dashboard_id,
                            "metadata": dashboard_meta,
                        },
                        "chart_meta": chart_meta,
                        "chart_detail": chart_detail,
                        "query_index": idx,
                    },
                )

    def _extract_dashboard_id(self, dashboard_url: str) -> str:
        stripped = (dashboard_url or "").strip()
        if stripped.isdigit():
            return stripped

        parsed = urlparse(stripped)
        if parsed.scheme and parsed.netloc:
            segments = [segment for segment in parsed.path.split("/") if segment]
            for segment in reversed(segments):
                if segment.isdigit():
                    return segment

            query_params = parse_qs(parsed.query)
            for key in ("dashboard_id", "id"):
                values = query_params.get(key)
                if values:
                    return values[0]

        return stripped

    def _chart_title(
        self, chart_meta: Dict[str, Any], chart_detail: Optional[Dict[str, Any]], query_index: int = 0
    ) -> str:
        base_title = (chart_meta.get("slice_name") or "") if chart_meta else ""
        if not base_title and chart_detail:
            base_title = chart_detail.get("slice_name") or ""

        if query_index > 0:
            suffix = f" (query {query_index + 1})"
            return f"{base_title}{suffix}" if base_title else suffix.strip()

        return base_title

    def _chart_description(
        self, chart_meta: Optional[Dict[str, Any]], chart_detail: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        description = None
        if chart_meta:
            description = chart_meta.get("description")
        if not description and chart_detail:
            description = chart_detail.get("description")
        return description

    def _extract_query_context(self, chart_detail: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        raw_context = chart_detail.get("query_context")
        parsed_context = self._load_json_field(raw_context)
        if isinstance(parsed_context, dict):
            return parsed_context
        return None

    def _collect_sql_from_chart(
        self,
        chart_id: Union[str, int],
        query_context: Optional[Dict[str, Any]],
    ) -> List[str]:
        if not query_context:
            logger.debug(f"No query_context for chart {chart_id}")
            return []

        payload = dict(query_context)
        payload.setdefault("result_format", "json")
        payload.setdefault("result_type", "query")

        try:
            response_data = self._request_json("POST", "chart/data", json=payload)
        except SupersetAdaptorError as exc:
            raise SupersetAdaptorError(f"chart/data failed for {chart_id}: {exc}") from exc

        sqls: List[str] = []
        results = response_data.get("result")

        if isinstance(results, list):
            for block in results:
                self._append_sql_from_block(block, sqls)
        elif isinstance(results, dict):
            self._append_sql_from_block(results, sqls)
        return sqls

    def _append_sql_from_block(self, block: Dict[str, Any], sqls: List[str]) -> None:
        queries = block.get("queries") or []
        if queries and isinstance(queries, list):
            for query_def in queries:
                sql_text = query_def.get("query")
                if sql_text:
                    sqls.append(sql_text.strip())

        if sql_text := block.get("query"):
            sqls.append(sql_text.strip())

    def _get_dashboard(self, dashboard_id: Union[str, int]) -> Dict[str, Any]:
        data = self._request_json("GET", f"dashboard/{dashboard_id}")
        if "result" in data and isinstance(data["result"], dict):
            return data["result"]
        return data

    def _get_dashboard_charts(self, dashboard_id: Union[str, int]) -> List[Dict[str, Any]]:
        data = self._request_json("GET", f"dashboard/{dashboard_id}/charts")
        charts = data.get("result", data)
        if not isinstance(charts, list):
            raise SupersetAdaptorError(f"Unexpected charts payload: {charts}")
        return charts

    def _get_chart(self, chart_id: Union[str, int]) -> Dict[str, Any]:
        data = self._request_json("GET", f"chart/{chart_id}")
        chart = data.get("result", data)
        return chart

    def _load_json_field(self, value: Any) -> Any:
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str) and value:
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                logger.debug(f"Failed to decode JSON field: {value[:128]}")
        return None

    def _request_json(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        response = self._request(method, endpoint, **kwargs)
        try:
            return response.json()
        except json.JSONDecodeError as exc:
            raise SupersetAdaptorError(f"Invalid JSON response for {endpoint}: {exc}") from exc

    def _request(self, method: str, endpoint: str, require_auth: bool = True, **kwargs) -> httpx.Response:
        url = f"{self._api_base}/{endpoint.lstrip('/')}"
        headers = kwargs.pop("headers", {})
        if require_auth:
            self._ensure_authenticated()
            if self._auth_header_value:
                if self.auth_type == AuthType.LOGIN:
                    headers.setdefault("Authorization", self._auth_header_value)
                else:
                    # For test
                    headers.setdefault("x-csrftoken", self._auth_header_value)

        try:
            response = self._client.request(method, url, headers=headers, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            raise SupersetAdaptorError(
                f"Superset API {method} {endpoint} failed with {exc.response.status_code}: {exc.response.text}"
            ) from exc
        except httpx.HTTPError as exc:
            raise SupersetAdaptorError(f"Superset API {method} {endpoint} failed: {exc}") from exc

    def _ensure_authenticated(self) -> None:
        if self._auth_header_value and self._token_expiration and time.time() < self._token_expiration:
            return
        self._authenticate()

    def _authenticate(self) -> None:
        if self.auth_type == AuthType.LOGIN:
            if not isinstance(self.auth_params, dict):
                raise SupersetAdaptorError("auth_params must be a mapping when using LOGIN auth_type")
            payload = {
                "username": self.auth_params.get("username"),
                "password": self.auth_params.get("password"),
                "provider": self.auth_params.get("provider", "db"),
                "refresh": True,
            }
            try:
                response = self._request("POST", "security/login", require_auth=False, json=payload)
            except SupersetAdaptorError as exc:
                raise SupersetAdaptorError(f"Authentication failed: {exc}") from exc

            data = response.json()
            token_payload = data.get("result", data)
            access_token = token_payload.get("access_token")
            token_type = token_payload.get("token_type", "Bearer")
            expires_in = token_payload.get("expires_in")

            if not access_token:
                raise SupersetAdaptorError("Superset login response missing access_token")

            self._auth_header_value = f"{token_type} {access_token}".strip()
            if isinstance(expires_in, (int, float)) and expires_in > 0:
                self._token_expiration = time.time() + expires_in - 60
            else:
                self._token_expiration = time.time() + 3600
        else:
            self._auth_header_value = str(self.auth_params)
            # Treat API keys as long-lived credentials
            self._token_expiration = time.time() + 365 * 24 * 60 * 60


if __name__ == "__main__":
    adapter = SupersetAdaptor(
        base_url="https://superset.datatest.ch",
        auth_type=AuthType.API_KEY,
        auth_params="Ijg1MzQwM2UzYzRlMjkzM2E5NWM3ODc4NDAyNTQ1M2FhY2ZkZTA1YzQi.aThGsg.6PYthEHMyeBP7QpWQPbhYCLKRY0",
    )
    for sql_pair in adapter.parse_sql_pair("https://superset.datatest.ch/superset/dashboard/8/"):
        print(sql_pair)
