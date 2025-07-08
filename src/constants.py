import os
import urllib
import streamlit as st
from collections import namedtuple

from dotenv import load_dotenv

# load_dotenv("../.env")

# BASE_URL = os.getenv("FIDDLER_BASE_URL")
BASE_URL = st.secrets["FIDDLER_BASE_URL"]
# TOKEN = os.getenv("FIDDLER_ACCESS_TOKEN")
TOKEN = st.secrets["FIDDLER_ACCESS_TOKEN"]
APPLICATION_ID = st.secrets["FIDDLER_APPLICATION_ID"]

CUSTOM_METRICS_URL = urllib.parse.urljoin(
    BASE_URL, "/v3/models/{model_id}/custom-metrics"
)

METRICS_URL = urllib.parse.urljoin(BASE_URL, "/v3/models/{model_id}/metrics")
QUERY_URL = urllib.parse.urljoin(BASE_URL, "/v3/queries")

GET_HEADER = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "*/*",
}

DATA = "data"
ITEMS = "items"
ID = "id"
NAME = "name"
DESCRIPTION = "description"
DEFINITION = "definition"

# OpenTelemetry
SERVICE_NAME = "fdl-chat"
# OTEL_ENDPOINT = "http://localhost:4318/v1/traces"
OTEL_ENDPOINT = "https://preprod.cloud.fiddler.ai/v1/traces"

## OpenTelemetry Span Names
CHAT_LOOP = "chat_loop"
USER_INPUT = "user_input"
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"
CONTENT = "content"
SYSTEM_PROMPT = "system_prompt"
AI_RESPONSE = "ai_response"

LLM_RESPONSE = "llm_response"
LLM_TOOL_RESPONSE = "llm_tool_response"

TOOL_CHOICE = "tool_choice"
TOOL_CALL = "tool_call"
TOOL_CALL_ID = "tool_call_id"
TOOL_CALL_NAME = "tool_call_name"
TOOL_CALL_ARGS = "tool_call_args"
TOOL_CALL_RESULTS = "tool_call_results"

OTEL_EXP = "OTEL_EXP"
OTEL_EXP_CONSOLE = "console"
OTEL_EXP_COLLECTOR = "collector"

PLAN_N_SOLVE = "plan_n_solve"
PLAN = "plan"

PNS_GENERATE_PLAN = "pns_generate_plan"
PNS_GENERATE_PLAN_RESPONSE = "pns_generate_plan_response"

PNS_EXEC_PLAN = "pns_exec_plan"
PNS_EXEC_PLAN_RESPONSE = "pns_exec_plan_response"

PNS_TOOL_LLM_RESPONSE = "pns_tool_llm_response"
PNS_RESULT = "pns_result"

SESSION_ID = "session_id"

ST_TITLE = "FiddleBot"
ST_MESSAGES = "messages"
ST_ROLE = "role"
ST_CONTENT = "content"

ST_FAVICON_PATH = "./assets/fdl_black_logo.svg"
ST_ICON_PATH = "./assets/fdl_black_logo.svg"
ST_CONFIG_PATH = "./config.yaml"

ST_FDL_ROLE = "FiddleBot"
ST_USER_ROLE = "user"
ST_CREDENTIALS = "credentials"
ST_COOKIE = "cookie"
ST_NAME = "name"
ST_KEY = "key"
ST_EXPIRY_DAYS = "expiry_days"
ST_RIGHT_ALIGN = """<div style="text-align: right;">{text}</div>"""
ST_USERNAMES = "usernames"
ST_ADMIN = "admin"
ST_EMAIL = "email"
ST_NAME = "name"
ST_PASSWORD = "password"

ST_USER_NAME = "USER_NAME"
ST_USER_EMAIL = "USER_EMAIL"
ST_USER_PASSWORD = "USER_PASSWORD"

ST_EXPIRY_DAYS = "expiry_days"
ST_KEY = "key"

ST_COOKIE_EXPIRY_DAYS = "COOKIE_EXPIRY_DAYS"
ST_COOKIE_KEY = "COOKIE_KEY"
ST_COOKIE_NAME = "COOKIE_NAME"

ST_AUTH_STATUS = "authentication_status"
ST_CONVERSATION_HISTORY = "conversation_history"

MCP_SERVER_NAME = "fdl-server"
ST_SESSION_ID = "_session_id"

ST_FDL_ENV_MESSAGE = "FiddleBot Observability Environment : preprod.cloud.fiddler.ai"

MAX_TOKENS = 4096

POST_HEADER = {
    "Authorization": f"Bearer {TOKEN}",
    "Content-Type": "application/json",
    "X-Integration": "FiddleBot",
}

COLUMN_INFO = "ColumnInfo"

METRICS_URL = urllib.parse.urljoin(BASE_URL, "/v3/models/{model_id}/metrics")

TYPE = "type"
COLUMNS = "columns"
PERFORMANCE = "performance"

PERF_METRIC = "PerfMetric"

PerfMetricInfo = namedtuple(PERF_METRIC, [ID, NAME])
ColumnInfo = namedtuple(COLUMN_INFO, [ID, NAME])

DATA = "data"
METRICS = "metrics"

# Constants for main payload keys
PROJECT_ID = "project_id"
TIME_COMPARISON = "time_comparison"
QUERY_TYPE = "query_type"
FILTERS = "filters"
QUERIES = "queries"

# Constants for filters keys
FILTERS_TIME_LABEL = "time_label"
FILTERS_TIME_RANGE = "time_range"
FILTERS_TIME_ZONE = "time_zone"
FILTERS_BIN_SIZE = "bin_size"

# Constants for time_range keys
TIME_RANGE_START_TIME = "start_time"
TIME_RANGE_END_TIME = "end_time"

# Constants for queries item keys
QUERY_KEY = "query_key"
MODEL_ID = "model_id"
BASELINE_ID = "baseline_id"
METRIC = "metric"
CATEGORIES = "categories"
SEGMENT_ID = "segment_id"
SEGMENT = "segment"
VIZ_TYPE = "viz_type"

# Constants for segment keys
SEGMENT_ID_KEY = "id"
SEGMENT_DEFINITION = "definition"

PREV_DAY = "Previous Day"
PREV_WEEK = "Previous Week"

VIZ_TYPE_LINE = "line"
MONITORING = "MONITORING"

QUERY_PARAM = {
    QUERY_KEY: None,
    MODEL_ID: None,
    BASELINE_ID: None,
    METRIC: None,
    COLUMNS: [],
    CATEGORIES: [],
    SEGMENT: {},
    VIZ_TYPE: None,
}

QUERY_PAYLOAD = {
    PROJECT_ID: None,
    QUERY_TYPE: MONITORING,
    FILTERS: {
        FILTERS_TIME_LABEL: None,
        FILTERS_TIME_RANGE: {
            TIME_RANGE_START_TIME: None,
            TIME_RANGE_END_TIME: None,
        },
        FILTERS_TIME_ZONE: None,
        FILTERS_BIN_SIZE: None,
    },
    QUERIES: [],
}

WEEK = "Week"
DAYS_7 = "7d"
DAY = "Day"
REQUIRES_BASELINE = "requires_baseline"
REQUIRES_CATEGORIES = "requires_categories"


DATE = "date"
RESULTS = "results"
VALUE = "value"


## Open Telemetry Constants

# LLM Attributes
ATTR_SYSTEM_PROMPT = "gen_ai.llm.input.system"
ATTR_USER_PROMPT = "gen_ai.llm.input.user"
ATTR_OUTPUT = "gen_ai.llm.output"
ATTR_CONTEXT = "gen_ai.llm.context"

# Tool Attributes
TOOL_ATTR_NAME = "gen_ai.tool.name"
TOOL_ATTR_INPUT = "gen_ai.tool.input"
TOOL_ATTR_OUTPUT = "gen_ai.tool.output"

# Common Attributes
SPAN_TYPE = "fiddler.span.type"
AGENT_NAME = "gen_ai.agent.name"
AGENT_ID = "gen_ai.agent.id"
CONV_ID = "gen_ai.conversation.id"

OTEL_HEADER = {
    "authorization": f"Bearer {TOKEN}",
    "fiddler-application-id": APPLICATION_ID,
}

OTEL_RESOURCE_APP_ID = "application.id"
SPAN_TYPE_LLM = "llm"
SPAN_TYPE_TOOL = "tool"
SPAN_TYPE_CHAIN = "chain"
SPAN_TYPE_OTHER = "other"

AGENT_NAME = "gen_ai.agent.name"
AGENT_ID = "gen_ai.agent.id"
CONVERSATION_ID = "gen_ai.conversation.id"

CONV_AGENT = "conversation_agent"
PNS_AGENT = "pns_agent"
QUERY_AGENT = "query_agent"

## Performance Query Agent
QUERY_AGENT = "query_agent"
HAS_PERFORMANCE_SPAN = "has_performance_span"
MODEL_PROJ_SPAN = "model_proj_span"
START_END_DATE_SPAN = "start_end_date_span"
