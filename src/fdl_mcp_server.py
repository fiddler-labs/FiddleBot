import os
import requests
import fiddler as fdl

from typing import Any
from rich import print
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from datetime import datetime, timedelta

import constants

load_dotenv()

MODEL_ID = "model_id"
PROJECT_ID = "project_id"
ID = "id"
_RESP = "_resp"

BASE_URL = os.getenv("FIDDLER_BASE_URL")
TOKEN = os.getenv("FIDDLER_ACCESS_TOKEN")

mcp = FastMCP("fdl-server")
fdl.init(url=BASE_URL, token=TOKEN)
print("Fiddler Client Initialised")


## Internal Function
def get_model(project_name: str, model_name: str):
    """Get a model given project name and model name

    Args:
        project_name: Name of project
        model_name: Name of model
    """
    project = fdl.Project.from_name(name=project_name)
    model = fdl.Model.from_name(name=model_name, project_id=project.id)
    return model


@mcp.tool()
def list_all_projects() -> list[str]:
    """
    List the names of all projects in the organisation
    """
    print("Listing all projects")
    project_names = []
    for project in fdl.Project.list():
        project_names.append(str(project.name))
    return project_names


@mcp.tool()
def list_models_in_project(project_name: str) -> list[str]:
    """
    List out all model names associated with a project

    Args:
        project_name: Name of the project
    """
    project = fdl.Project.from_name(name=project_name)
    model_names = []
    for model in project.models:
        model_names.append(str(model.name))
    return model_names


@mcp.tool()
def get_model_schema(project_name: str, model_name: str):
    """Get the schema for a given model

    Args:
        project_name: Name of the project
        model_name: Name of the model
    """
    model = get_model(project_name=project_name, model_name=model_name)
    model_schema = model.schema.json()
    return model_schema


@mcp.tool()
def get_model_spec(project_name: str, model_name: str):
    """Get model specs given a project name and model name
    Model Specs tell Fiddler what features/columns are inputs, outputs, targets/labels and metadata

    Args:
        project_name: Name of the project
        model_name: Name of the model
    """
    model = get_model(project_name=project_name, model_name=model_name)
    model_spec = model.spec.json()
    return model_spec


@mcp.tool()
def list_alertrules_for_model(project_name: str, model_name: str):
    """Get Alert Rules for a given project and model
    An alert rule is used to setup checks and notification rules on a model's health and performance, over time.

    Args:
        project_name: Name of the project
        model_name: Name of the model
    """
    model = get_model(project_name, model_name)
    alert_rules = fdl.AlertRule.list(model_id=model.id)
    rules = []
    for rule in alert_rules:
        print(rule)
        rule = vars(rule)
        del rule[MODEL_ID]
        del rule[PROJECT_ID]
        del rule[_RESP]
        rules.append(rule)
    return rules


@mcp.tool()
def list_triggered_alerts_for_rule(
    project_name: str, model_name: str, alert_rule_id: str, days: int = 1
):
    """Get triggered alerts for a given model in the past X days

    Args:
        project_name: Name of the project
        model_name: Name of the model
        alert_rule_id: ID of the alert rule
        days: Number of days to look back (default 1)
    """
    model = get_model(project_name, model_name)
    print("################")
    print(f"Model: {model.name}")
    print("################")

    # Get alerts triggered in the last X days
    alert_records = fdl.AlertRecord.list(
        alert_rule_id=alert_rule_id,
        start_time=datetime.now() - timedelta(days=days),
        end_time=datetime.now(),
    )

    records = []
    for record in alert_records:
        record = vars(record)
        # Remove internal fields
        del record[MODEL_ID]
        del record[PROJECT_ID]
        del record[ID]
        del record[_RESP]
        records.append(record)

    return records


@mcp.tool()
def list_all_model_custom_metrics(project_name: str, model_name: str):
    """List all custom metrics for a given model
    Custom metrics are metrics that are not part of the model's schema, but are calculated by the model.
    """
    model = get_model(project_name, model_name)
    model_id = str(model.id)
    url = constants.CUSTOM_METRICS_URL.format(model_id=model_id)
    response = requests.get(url, headers=constants.GET_HEADER)
    custom_metrics = []
    resp_json = response.json()
    items = resp_json[constants.DATA][constants.ITEMS]
    for item in items:
        name = item[constants.NAME]
        description = item[constants.DESCRIPTION]
        definition = item[constants.DEFINITION]
        metric_id = item[constants.ID]
        custom_metrics.append(
            {
                constants.ID: metric_id,
                constants.NAME: name,
                constants.DESCRIPTION: description,
                constants.DEFINITION: definition,
            }
        )
    return custom_metrics


if __name__ == "__main__":
    mcp.run(transport="stdio")
