import os
import pdb
import requests
import datetime
import pandas as pd
import fiddler as fdl

from uuid import uuid4
from copy import deepcopy
from pydantic import BaseModel
from datetime import timedelta
from langfuse.openai import OpenAI, AsyncOpenAI


import utils
import constants
import fdl_tracer

MODEL_ID = "model_id"
PROJECT_ID = "project_id"
ID = "id"
_RESP = "_resp"

BASE_URL = os.getenv("FIDDLER_BASE_URL")
TOKEN = os.getenv("FIDDLER_ACCESS_TOKEN")

HAS_PERFORMANCE_QUERY_PROMPT = "{message}\nHas the user asked to query performance metrics for a given model? Think through step by step before giving your answer."

EXTRACT_PROJECT_AND_MODEL_NAME_PROMPT = "{message}\nExtract the project and model name that the user wants to query performance metrics for. Think through step by step before giving your answer."

EXTRACT_START_END_DATE_PROMPT = "{message}\nExtract the start and end dates that the user wants to query performance metrics for. Think through step by step before giving your answer. The dates should be in YYYY-MM-DD format."


class HasPerformanceQuery(BaseModel):
    reasoning: str
    has_performance_query: bool


class ProjectModel(BaseModel):
    reasoning: str
    project_name: str
    model_name: str


class QueryDates(BaseModel):
    reasoning: str
    start_date: str
    end_date: str


fdl.init(url=BASE_URL, token=TOKEN)


def get_model_metrics(project_name: str, model_name: str):
    """Get all metrics associated with the model"""
    model = utils.get_model(project_name, model_name)
    metrics_url = constants.METRICS_URL.format(model_id=model.id)
    response = requests.get(metrics_url, headers=constants.GET_HEADER)
    return response.json()


def extract_model_columns(all_metrics: dict):
    """Extract the name and IDs of all columns in the model"""
    column_details = all_metrics["data"]["columns"]
    column_id_names = []
    for column in column_details:
        column_id_names.append(
            constants.ColumnInfo(column[constants.ID], column[constants.NAME])
        )
    return column_id_names


def extract_performance_metrics(all_metrics: dict):
    """Extract performance metrics associated with the model"""
    raw_details = all_metrics[constants.DATA][constants.METRICS]
    performance_metrics = []
    for details in raw_details:
        if details[constants.TYPE] == constants.PERFORMANCE:
            performance_metrics.append(
                constants.PerfMetricInfo(details[constants.ID], details[constants.NAME])
            )
    return performance_metrics


def get_current_time() -> str:
    """Get the current time in YYYY-MM-DD format"""
    return datetime.now().strftime("%Y-%m-%d")


def get_days_ago(num_days: int) -> str:
    """Convert a number of days to a datetime string in YYYY-MM-DD format"""
    return (datetime.now() - timedelta(days=num_days)).strftime("%Y-%m-%d")


def parse_date(date_str: str):
    """Parse the given date"""
    date = datetime.datetime.fromisoformat(date_str).date()
    return date


def parse_performance_response(resp_json: dict):
    """Parse performance response and return a dataframe with values"""
    data = {}

    df_skeleton = {
        constants.DATE: [],
        constants.VALUE: [],
    }
    for query_key, result in resp_json[constants.DATA][constants.RESULTS].items():
        metric = result[constants.METRIC]
        values = result[constants.DATA]
        skeleton = deepcopy(df_skeleton)
        for value in values:
            date = parse_date(value[0])
            value = value[1]

            skeleton[constants.DATE].append(date)
            skeleton[constants.VALUE].append(value)

        data[metric] = pd.DataFrame(skeleton)

    return data


def get_performance_metrics(
    project_name: str, model_name: str, start_date: str, end_date: str
):
    """Get the performance metrics for a model, given the start and end dates"""
    utils.print("Getting performance metrics")
    model = utils.get_model(project_name, model_name)
    all_metrics = get_model_metrics(project_name, model_name)
    performance_metrics = extract_performance_metrics(all_metrics)
    query_filters = []
    for metric in performance_metrics:
        query_filter = deepcopy(constants.QUERY_PARAM)
        query_filter[constants.QUERY_KEY] = str(uuid4())
        query_filter[MODEL_ID] = str(model.id)
        query_filter[constants.BASELINE_ID] = ""
        query_filter[constants.METRIC] = metric.id
        query_filter[constants.VIZ_TYPE] = constants.VIZ_TYPE_LINE
        query_filters.append(query_filter)

    payload = deepcopy(constants.QUERY_PAYLOAD)
    payload[constants.PROJECT_ID] = str(model.project_id)
    payload[constants.FILTERS][constants.FILTERS_BIN_SIZE] = constants.DAY
    payload[constants.FILTERS][constants.FILTERS_TIME_LABEL] = constants.DAYS_7
    payload[constants.FILTERS][constants.FILTERS_TIME_RANGE][
        constants.TIME_RANGE_START_TIME
    ] = start_date
    payload[constants.FILTERS][constants.FILTERS_TIME_RANGE][
        constants.TIME_RANGE_END_TIME
    ] = end_date
    payload[constants.FILTERS][constants.FILTERS_TIME_ZONE] = "UTC"
    payload[constants.QUERIES] = query_filters

    response = requests.post(
        constants.QUERY_URL, json=payload, headers=constants.POST_HEADER
    )

    metrics_df = parse_performance_response(response.json())
    return metrics_df


class FdlQueries:
    """Fiddler Queries Client"""

    def __init__(self):
        self.MODEL_NAME = "gpt-4.1-mini"

        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.fdl_tracer = fdl_tracer.FdlTracer.get_instance(
            tracer_name=constants.SERVICE_NAME,
            otel_export=os.getenv(constants.OTEL_EXP),
        )
        self.tracer = self.fdl_tracer.get_tracer()

    def has_performance_query(self, conversation_history: list) -> bool:
        """Check if the conversation history contains a performance query"""
        user_message = conversation_history[-1][constants.CONTENT]
        user_query = utils.create_message(
            constants.USER_ROLE,
            HAS_PERFORMANCE_QUERY_PROMPT.format(message=user_message),
        )
        modified_conversation_history = conversation_history[:-1] + [user_query]

        response = self.client.chat.completions.parse(
            model=self.MODEL_NAME,
            messages=modified_conversation_history,
            max_tokens=constants.MAX_TOKENS,
            response_format=HasPerformanceQuery,
        )

        with self.tracer.start_as_current_span(
            constants.HAS_PERFORMANCE_SPAN
        ) as has_performance_span:
            has_performance_span.set_attribute(
                constants.AGENT_NAME, constants.QUERY_AGENT
            )
            has_performance_span.set_attribute(
                constants.SPAN_TYPE, constants.SPAN_TYPE_LLM
            )
            has_performance_span.set_attribute(constants.ATTR_USER_PROMPT, user_message)
            has_performance_span.set_attribute(
                constants.ATTR_OUTPUT,
                response.choices[0].message.parsed.has_performance_query,
            )

        return response.choices[0].message.parsed.has_performance_query

    def extract_project_and_model_name(
        self, conversation_history: list
    ) -> tuple[str, str]:
        """Extract the project and model name from the conversation history"""
        utils.print("Extracting project and model name")
        user_message = conversation_history[-1][constants.CONTENT]
        user_query = utils.create_message(
            constants.USER_ROLE,
            EXTRACT_PROJECT_AND_MODEL_NAME_PROMPT.format(message=user_message),
        )
        modified_conversation_history = conversation_history[:-1] + [user_query]
        response = self.client.chat.completions.parse(
            model=self.MODEL_NAME,
            messages=modified_conversation_history,
            max_tokens=constants.MAX_TOKENS,
            response_format=ProjectModel,
        )
        with self.tracer.start_as_current_span(
            constants.MODEL_PROJ_SPAN
        ) as model_proj_span:
            model_proj_span.set_attribute(constants.AGENT_NAME, constants.QUERY_AGENT)
            model_proj_span.set_attribute(constants.SPAN_TYPE, constants.SPAN_TYPE_LLM)
            model_proj_span.set_attribute(constants.ATTR_USER_PROMPT, user_message)
            simplified_output = f"Project: {response.choices[0].message.parsed.project_name}, Model: {response.choices[0].message.parsed.model_name}"
            model_proj_span.set_attribute(constants.ATTR_OUTPUT, simplified_output)

        # with self.tracer.start_as_current_span(constants.QUERY_AGENT) as query_agent:

        return (
            response.choices[0].message.parsed.project_name,
            response.choices[0].message.parsed.model_name,
        )

    def extract_start_end_date(self, conversation_history: list) -> tuple[str, str]:
        """Extract the start and end dates from the conversation history"""
        utils.print("Extracting start and end date")
        user_message = conversation_history[-1][constants.CONTENT]
        user_query = utils.create_message(
            constants.USER_ROLE,
            EXTRACT_START_END_DATE_PROMPT.format(message=user_message),
        )
        modified_conversation_history = conversation_history[:-1] + [user_query]
        response = self.client.chat.completions.parse(
            model=self.MODEL_NAME,
            messages=modified_conversation_history,
            max_tokens=constants.MAX_TOKENS,
            response_format=QueryDates,
        )
        with self.tracer.start_as_current_span(
            constants.START_END_DATE_SPAN
        ) as start_end_date_span:
            start_end_date_span.set_attribute(
                constants.AGENT_NAME, constants.QUERY_AGENT
            )
            start_end_date_span.set_attribute(
                constants.SPAN_TYPE, constants.SPAN_TYPE_LLM
            )
            start_end_date_span.set_attribute(constants.ATTR_USER_PROMPT, user_message)
            simplified_output = f"Start Date: {response.choices[0].message.parsed.start_date}, End Date: {response.choices[0].message.parsed.end_date}"
            start_end_date_span.set_attribute(constants.ATTR_OUTPUT, simplified_output)

        start_date = response.choices[0].message.parsed.start_date
        end_date = response.choices[0].message.parsed.end_date

        if start_date == "" or end_date == "":
            return None, None

        start_date = utils.validate_and_convert_date(start_date)
        end_date = utils.validate_and_convert_date(end_date)

        return start_date, end_date

    def get_performance_metrics(
        self, project_name: str, model_name: str, start_date: str, end_date: str
    ):
        """Get the performance metrics for a model, given the start and end dates"""
        utils.print("Getting performance metrics")
        return get_performance_metrics(project_name, model_name, start_date, end_date)
