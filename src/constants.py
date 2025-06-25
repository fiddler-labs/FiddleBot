import os
import urllib
import streamlit as st

from dotenv import load_dotenv

# load_dotenv("../.env")

# BASE_URL = os.getenv("FIDDLER_BASE_URL")
BASE_URL = st.secrets["FIDDLER_BASE_URL"]
# TOKEN = os.getenv("FIDDLER_ACCESS_TOKEN")
TOKEN = st.secrets["FIDDLER_ACCESS_TOKEN"]

CUSTOM_METRICS_URL = urllib.parse.urljoin(
    BASE_URL, "/v3/models/{model_id}/custom-metrics"
)

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
OTEL_ENDPOINT = "http://localhost:4318/v1/traces"

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
