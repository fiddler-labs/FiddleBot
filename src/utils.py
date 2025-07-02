import os

import requests
import pandas as pd
import fiddler as fdl
from copy import deepcopy
from rich import console, markdown
from datetime import datetime, timedelta

import constants

MODEL_ID = "model_id"
PROJECT_ID = "project_id"
ID = "id"
_RESP = "_resp"

BASE_URL = os.getenv("FIDDLER_BASE_URL")
TOKEN = os.getenv("FIDDLER_ACCESS_TOKEN")

fdl.init(url=BASE_URL, token=TOKEN)

TYPE = "type"
ROLE = "role"
CONTENT = "content"
FUNC_CALL_OUTPUT = "function_call_output"
TOOL_CALLS = "tool_calls"
TOOL_CALL_ID = "tool_call_id"
CALL_ID = "call_id"
OUTPUT = "output"
CONSOLE = console.Console()


def create_message(
    role: str, content: str, tool_calls: list = None, tool_call_id: str = None
) -> dict:
    """
    Create an OpenAI Message unit

    role: str
    content: str
    """
    if tool_calls is not None:
        return {ROLE: role, CONTENT: content, TOOL_CALLS: tool_calls}

    if tool_call_id is not None:
        return {ROLE: role, CONTENT: content, TOOL_CALL_ID: tool_call_id}
    else:
        return {ROLE: role, CONTENT: content}


def create_tool_message(call_id: str, output: str) -> dict:
    return {
        TYPE: FUNC_CALL_OUTPUT,
        CALL_ID: call_id,
        OUTPUT: output,
    }


def print(message: str):
    """Print to console"""
    md = markdown.Markdown(message)
    CONSOLE.print(md)


def validate_and_convert_date(date_str: str) -> str:
    """
    Validate YYYY-MM-DD format and convert to ISO datetime.

    Returns ISO datetime string or raises ValueError if invalid.
    """
    return datetime.strptime(date_str, "%Y-%m-%d").isoformat()


def get_model(project_name: str, model_name: str):
    """Get the model"""
    project = fdl.Project.from_name(name=project_name)
    model = fdl.Model.from_name(name=model_name, project_id=project.id)
    return model
