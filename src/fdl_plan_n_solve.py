import os
import pdb
import json
import asyncio
import streamlit as st

from copy import deepcopy
from fastmcp import Client
from typing import Optional
from dotenv import load_dotenv
from pydantic import BaseModel
from contextlib import AsyncExitStack
from mcp.client.stdio import stdio_client
from langfuse.openai import openai, AsyncOpenAI
from mcp import ClientSession, StdioServerParameters
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

import utils
import constants
import fdl_tracer
import fdl_fast_mcp

load_dotenv()

SERVER_PATH = "server_path"
MODEL_4o_MINI = "gpt-4o-mini"
MODEL_41_MINI = "gpt-4.1"
MODEL_04_MINI = "o4-mini"
MESSAGE = "message"
FUNC_CALL = "function_call"

FDL_PNS_SYSTEM_PROMPT = """Fiddler is the pioneer in AI Observability and Security, 
enabling organizations to build trustworthy and responsible AI systems. Our platform helps Data Science teams, 
MLOps engineers, and business stakeholders monitor, explain, analyze, and improve their AI deployments.

With Fiddler, teams can:
- Monitor performance of ML models and generative AI applications
- Protect your LLM and GenAI applications with Guardrails
- Analyze model behavior to identify issues and opportunities
- Improve AI systems through actionable insights

You are a helpful assistant who is very well versed in Fiddler and it's capabilities.
Users have already been authenticated and are logged in to Fiddler.

Return all the results completely. Do not truncate anything.
"""


FDL_PNS_MULTI_STEP_PROMPT = """You will be given a task to solve. Generate a list of tool calls to solve this task using the set of tools available.
Use as few tools as possible to solve the task.
Think through this step by step before giving your final output.
Output only the plan, which should just be a list of steps and nothing more.
Task to complete: {task}
"""

_FDL_PNS_MULTI_STEP_PROMPT = """
You are given a task to solve. Generate a plan that is instrumental in solving the task.
Think through this step by step before giving the plan. Keep the list of tools in mind. DO NOT MAKE ANY TOOL CALL.
Tasks  {task}
Having read the task, think through the steps to solve the task.
Task : {task}
"""

FDL_STEP_EXEC_PROMPT = (
    """Execute the following step by making the appropriate tool call. Step: {step}"""
)

FLD_RESULT_PROMPT = """The tasks have been executed. Extract the results from the previous steps and provide a clear and concise response."""

LLM_TASK_PROMPT = "llm_task_prompt"


class FiddlerPlan(BaseModel):
    reasoning: list[str]
    plan: list[str]


class FiddlerExecClient:
    """Fiddler Exec Client is a planning and decomposition tool that uses an MCP server to execute planned actions"""

    def __init__(
        self,
        server_script_path=None,
        system_prompt=FDL_PNS_SYSTEM_PROMPT,
        multi_step_prompt=FDL_PNS_MULTI_STEP_PROMPT,
        step_exec_prompt=FDL_STEP_EXEC_PROMPT,
    ):
        utils.print("Initialising FiddlerExecClient")
        self.MODEL_NAME = MODEL_41_MINI
        self.TYPE = "type"
        self.NAME = "name"
        self.DESC = "description"
        self.PARAMS = "parameters"
        self.ROLE = "role"
        self.CONTENT = "content"
        self.FUNC = "function"
        self.USER = "user"
        self.ASSISTANT = "assistant"
        self.SYST = "system"
        self.MAX_OUTPUT_TOKENS = 4096

        ###################################
        ## Prompts
        ###################################

        ## Need to add a lot of steps and documentation details to the system prompt
        self.SYSTEM_PROMPT = system_prompt
        self.MULTI_STEP_PROMPT = multi_step_prompt
        self.STEP_EXEC_PROMPT = step_exec_prompt

        self.syst_message = {
            self.ROLE: self.SYST,
            self.CONTENT: self.SYSTEM_PROMPT,
        }

        self.user_msg_template = {
            self.ROLE: self.USER,
            self.CONTENT: None,
        }

        self.conversation = [deepcopy(self.syst_message)]

        ## Number of words in the given conversation
        ## Conversation resets after crossing 50_000 words
        self.conv_length = len(self.SYSTEM_PROMPT.strip().split())

        self.planning_tool = [
            {
                "type": "function",
                "function": {
                    "name": "FiddlerExecClient",
                    "description": """Query the Fiddler platform to solve a single task. The task has to be in the third person.
                    The task has to be related to one or more models, projects or alerts that exists on the Fiddler platform.
                    Describe the task in great detail. Provide all relevant information related to the task.
                    Think this through step by step before generating the task prompt.
                    """,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "llm_task_prompt": {
                                "type": "string",
                                "description": "The task prompt to be planned and solved",
                            }
                        },
                        "required": ["llm_task_prompt"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        ]

        ###################################
        ## MCP Server Details
        ###################################

        if server_script_path is not None:
            self.server_script_path = server_script_path
            self.session: Optional[ClientSession] = None
            self.exit_stack = AsyncExitStack()
        else:
            self.server_script_path = None
            utils.print("Using In-Memory MCP Server")
            self.mcp_server = fdl_fast_mcp.FdlMCP.get_instance()
            self.mcp_client = self.mcp_server.get_client()

        ###################################
        ## LLM Client Details
        ###################################
        self.oai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.available_tools = None

        ###################################
        ## OTEL Details
        ###################################

        otel_exp = os.getenv(constants.OTEL_EXP)
        self.fdl_tracer = fdl_tracer.FdlTracer.get_instance(
            tracer_name=constants.SERVICE_NAME, otel_export=otel_exp
        )
        self.tracer = self.fdl_tracer.get_tracer()

        ## This is set in each session
        self.session_id = None

    async def connect_to_server(self):
        """Connect to MCP Server"""
        if self.server_script_path is not None:
            command = "python"
            utils.print("Loading Server Params")
            server_params = StdioServerParameters(
                command=command, args=[self.server_script_path], env=None
            )

            utils.print("Loading stdio_transport")
            stdio_transport = await self.exit_stack.enter_async_context(
                stdio_client(server_params)
            )

            utils.print("Entering Async Context")
            self.stdio, self.write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(self.stdio, self.write)
            )

            utils.print("Initialising Session")
            await self.session.initialize()

            utils.print("Get list of tools")
            mcp_tools_list = await self.session.list_tools()
            mcp_tools_list = mcp_tools_list.tools
        else:
            async with self.mcp_client:
                mcp_tools_list = await self.mcp_client.list_tools()

        self.available_tools = []
        for tool in mcp_tools_list:
            self.available_tools.append(
                {
                    self.TYPE: self.FUNC,
                    self.NAME: tool.name,
                    self.DESC: tool.description,
                    self.PARAMS: tool.inputSchema,
                }
            )

        utils.print(
            f"\nConnected to server with tools : {[tool[self.NAME] for tool in self.available_tools]}",
        )

    def list_tools(self):
        """List the tool plan_n_solve"""
        return self.planning_tool

    async def get_llm_response(self, messages: list[dict]) -> str:
        """Get a response from the LLM
        Args:
            messages: list[dict]
            use_tools: bool
        Returns:
            response: OpenAI response object
        """
        response = await self.oai_client.responses.create(
            model=self.MODEL_NAME,
            input=messages,
            tools=self.available_tools,
        )
        return response

    async def generate_plan(self, natural_language_task: str) -> str:
        """Generate a plan to solve the task"""
        utils.print(f"Generating plan for task: {natural_language_task}")
        system_message = utils.create_message(self.SYST, self.SYSTEM_PROMPT)
        user_message = utils.create_message(
            self.USER,
            self.MULTI_STEP_PROMPT.format(task=natural_language_task),
        )
        messages = [system_message, user_message]
        with self.tracer.start_as_current_span(
            constants.PNS_GENERATE_PLAN
        ) as gen_plan_span:
            response = await self.get_llm_response(messages)
            gen_plan_span.set_attribute(constants.AGENT_NAME, constants.PNS_AGENT)
            gen_plan_span.set_attribute(constants.SPAN_TYPE, constants.SPAN_TYPE_LLM)
            gen_plan_span.set_attribute(
                constants.ATTR_USER_PROMPT, user_message[constants.CONTENT]
            )
            gen_plan_span.set_attribute(constants.ATTR_OUTPUT, response.output_text)

        utils.print(f"Plan: {response.output_text}")
        steps_list = response.output_text.split("\n")
        steps_list = [step.strip() for step in steps_list]
        return steps_list

    async def execute_plan(self, steps_list: list[str]) -> str:
        """Execute the plan"""
        ## Each step needs the following
        # 1. Call the LLM to get a specific tool call response.
        # 2. Call the tool using the MCP Server
        # 3. Parse both previous steps into an LLM to generate a text output.
        # 4. The output of step 3 is fed as input to the next task that needs to be done.
        utils.print(f"Executing plan: {steps_list}")
        system_message = utils.create_message(self.SYST, self.SYSTEM_PROMPT)
        messages = [system_message]
        for step in steps_list:
            ## Call LLM to get a specific tool call response.

            ## Change the system prompt and user prompt
            user_message = utils.create_message(
                self.USER, self.STEP_EXEC_PROMPT.format(step=step)
            )
            messages.append(user_message)
            # Get tool call response from LLM
            tool_response = await self.get_llm_response(messages)

            for tool_call in tool_response.output:
                if tool_call.type != FUNC_CALL:
                    continue

                messages.append(tool_call)
                tool_name = tool_call.name
                tool_args = json.loads(tool_call.arguments)

                with self.tracer.start_as_current_span(
                    constants.TOOL_CALL
                ) as tool_call_span:
                    ## Call the tool via MCP server
                    utils.print(f"Tool call: {tool_call.name}")
                    # tool_result = await self.session.call_tool(tool_name, tool_args)

                    async with self.mcp_client:
                        tool_result = await self.mcp_client.call_tool(
                            tool_name, tool_args
                        )
                    # tool_call_span.set_attribute(
                    # constants.TOOL_CALL_ID, tool_call.call_id
                    # )
                    tool_call_span.set_attribute(
                        constants.AGENT_NAME, constants.PNS_AGENT
                    )
                    tool_call_span.set_attribute(
                        constants.SPAN_TYPE, constants.SPAN_TYPE_TOOL
                    )
                    tool_call_span.set_attribute(constants.TOOL_ATTR_NAME, tool_name)
                    tool_call_span.set_attribute(
                        constants.TOOL_ATTR_INPUT, tool_call.arguments
                    )
                    if (
                        len(tool_result.content) == 0
                        or len(tool_result.content[0].text) == 0
                    ):
                        tool_result_text = f"No Resuls from {tool_name}"
                    else:
                        tool_result_text = tool_result.content[0].text
                    tool_call_span.set_attribute(
                        constants.TOOL_ATTR_OUTPUT, tool_result_text
                    )

                ## Generate natural language response from tool result
                result_message = utils.create_tool_message(
                    tool_call.call_id, tool_result_text
                )
                messages.append(result_message)

                with self.tracer.start_as_current_span(
                    constants.PNS_TOOL_LLM_RESPONSE
                ) as tool_llm_response_span:
                    response = await self.get_llm_response(messages)
                    tool_llm_response_span.set_attribute(
                        constants.AGENT_NAME, constants.PNS_AGENT
                    )
                    tool_llm_response_span.set_attribute(
                        constants.SPAN_TYPE, constants.SPAN_TYPE_LLM
                    )
                    tool_llm_response_span.set_attribute(
                        constants.ATTR_OUTPUT, response.output_text
                    )

                messages.append(
                    utils.create_message(self.ASSISTANT, response.output_text)
                )

        result_message = utils.create_message(
            self.USER,
            FLD_RESULT_PROMPT,
        )

        ## Do I log
        messages.append(result_message)
        response = await self.get_llm_response(messages)
        return response.output_text

    async def plan_n_solve(
        self, llm_task_prompt: str, otel_context: dict, session_id: str
    ) -> str:
        self.session_id = session_id
        ## Generate Plan to solve the task
        task = llm_task_prompt[LLM_TASK_PROMPT]
        utils.print(f"Planning and solving task: {task}")

        otel_context = TraceContextTextMapPropagator().extract(otel_context)

        ## Plan
        with self.tracer.start_as_current_span(
            constants.PLAN_N_SOLVE, context=otel_context
        ) as plan_n_solve_span:
            plan = await self.generate_plan(task)
            utils.print(f"Plan: {'\n'.join(plan)}")
            plan_n_solve_span.set_attribute(constants.AGENT_NAME, constants.PNS_AGENT)
            plan_n_solve_span.set_attribute(
                constants.SPAN_TYPE, constants.SPAN_TYPE_CHAIN
            )
            plan_n_solve_span.set_attribute(constants.ATTR_USER_PROMPT, task)
            plan_n_solve_span.set_attribute(constants.ATTR_OUTPUT, "\n".join(plan))

        ## Solve
        with self.tracer.start_as_current_span(
            constants.PNS_EXEC_PLAN, context=otel_context
        ) as exec_plan_span:
            result = await self.execute_plan(plan)
            exec_plan_span.set_attribute(constants.AGENT_NAME, constants.PNS_AGENT)
            exec_plan_span.set_attribute(constants.SPAN_TYPE, constants.SPAN_TYPE_CHAIN)
            exec_plan_span.set_attribute(constants.ATTR_USER_PROMPT, "\n".join(plan))
            exec_plan_span.set_attribute(constants.ATTR_OUTPUT, result)

        self.session_id = None
        return result

    async def cleanup(self):
        """Cleanup the MCP server"""
        openai.flush_langfuse()
        if self.server_script_path is not None:
            await self.exit_stack.aclose()
