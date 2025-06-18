import pdb

import os
import sys
import json
import asyncio
import argparse
from rich import print
from copy import deepcopy
from typing import Optional
from pydantic import BaseModel
from rich.console import Console
from rich.markdown import Markdown
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from dotenv import load_dotenv
from openai import OpenAI

import utils

SERVER_PATH = "server_path"
MODEL_4o_MINI = "gpt-4o-mini"
MODEL_41_MINI = "gpt-4.1-mini"
MESSAGE = "message"
FUNC_CALL = "function_call"


class PlanningSteps(BaseModel):
    steps: list[str]


class FiddlerMCPClient:
    def __init__(self, syst_prompt="You are a helpful assistant."):
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

        ## Prompts
        self.PLANNING_TEMPLATE = """You will be given a user query. 
        Break down the query into a set of steps that 
        need to be solved sequentially in order to arrive at the answer. 
        Generate this plan as a list of steps. 
        Each step can be a tool call.
        
        Query : {query}
        """

        ## Context keys
        self.FDL_PROJ_NAME = "project_name"
        self.FDL_MODEL_NAME = "model_name"
        self.ALERT_RULES = "alert_rules"
        self.ALERT_RECORDS = "alert_records"

        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.oai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.available_tools = None

        self.syst_message = {
            self.ROLE: self.SYST,
            self.CONTENT: syst_prompt,
        }

        self.user_msg_template = {
            self.ROLE: self.USER,
            self.CONTENT: None,
        }

        ## Conversation history
        self.conversation = [deepcopy(self.syst_message)]

        ## Number of words in the given conversation
        ## Conversation resets after crossing 50_000 words
        self.conv_length = len(syst_prompt.strip().split())
        self.console = Console()

        self.context = {
            self.FDL_PROJ_NAME: None,
            self.FDL_MODEL_NAME: None,
            self.ALERT_RULES: None,
            self.ALERT_RECORDS: None,
        }

    def create_message(self, role: str, content: str):
        """Create a user message"""
        user_message = deepcopy(self.user_msg_template)
        user_message[self.ROLE] = role
        user_message[self.CONTENT] = content
        return user_message

    async def connect_to_server(self, server_script_path: str):
        """Connect to an MCP Server
        Args:
            server_script_path: Path to the server script
        """
        command = "python"
        utils.print("Loading Server Params")
        server_params = StdioServerParameters(
            command=command, args=[server_script_path], env=None
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
        self.available_tools = []
        for tool in mcp_tools_list.tools:
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

    def plan_steps(self, query: str):
        """Plan a set of steps that need to be solved in order to solve the given user query"""
        # user_message = utils.create_message(self.USER, query)
        planning_query = self.PLANNING_TEMPLATE.format(query=query)

        planning_message = utils.create_message(self.USER, planning_query)

        history = deepcopy(self.conversation)

        history.append(planning_message)

        llm_response = self.oai_client.beta.chat.completions.parse(
            model=self.MODEL_NAME,
            messages=history,
            # tools=self.available_tools,
            # tool_choice="auto",
            response_format=PlanningSteps,
        )

        steps = llm_response.choices[0].message.parsed
        breakpoint()

    async def plan_and_solve(self, query):
        """Plan and get the response from LLM"""
        steps = self.plan_steps(query)
        results = []
        for step in steps:
            result = self.process_step(step, results)
        final_result = self.process_results(results)
        return final_result

    async def get_lm_response(self):
        """Get the response from the LLM"""
        response = self.oai_client.responses.create(
            model=self.MODEL_NAME,
            input=self.conversation,
            max_output_tokens=self.MAX_OUTPUT_TOKENS,
            tools=self.available_tools,
        )

        final_response = []

        for output in response.output:
            if output.type == MESSAGE:
                ## Check if the model refuses
                final_response.append(output.content[0].text)
            elif output.type == FUNC_CALL:
                tool_name = output.name
                tool_args = json.loads(output.arguments)
                utils.print(f"Calling {tool_name} with {tool_args}")

                ## Update context if project_name or model_name are in tool args
                if self.FDL_PROJ_NAME in tool_args:
                    self.context[self.FDL_PROJ_NAME] = tool_args[self.FDL_PROJ_NAME]
                    self.context[self.FDL_MODEL_NAME] = None
                if self.FDL_MODEL_NAME in tool_args:
                    self.context[self.FDL_MODEL_NAME] = tool_args[self.FDL_MODEL_NAME]

                ## Calling the tool to get the results
                ## The result is passed to the LLM to get another response
                ## The response from the LLM with the tool call results are
                ## Persisted in the conversation thread
                result = self.session.call_tool(tool_name, tool_args)

                self.conversation.append(
                    {
                        self.ROLE: self.ASSISTANT,
                        self.CONTENT: result.model_dump_json(),
                    }
                )

                ## Not allowing tool call here to force a text output here
                response = self.oai_client.responses.create(
                    model=self.MODEL_NAME,
                    input=self.conversation,
                    max_output_tokens=self.MAX_OUTPUT_TOKENS,
                )
                ## Removing the tool call result
                self.conversation.pop()

                final_response.append(response.output_text)
            else:
                return "Failed to parse LLM output"
        return "\n".join(final_response)

    async def process_query(self, query: str) -> str:
        """Process a given query. Use tools if necessary"""
        response_text = await self.plan_and_solve(query)
        self.add_to_conversation(self.ASSISTANT, response_text)

        utils.print(f"\nAssistant : {response_text}")

    def add_to_conversation(self, role: str, query: str):
        """Adds the query to the current conversation"""
        query_len = len(query.strip().split())
        self.conv_length += query_len
        message = utils.create_message(role, query)
        self.add_message_to_conversation(message)

    def add_message_to_conversation(self, message: dict):
        """Add a message to the conversation"""
        self.conversation.append(message)

    async def chat_loop(self):
        """Interactive Chat Loop"""
        utils.print("MCP Server is up and running")
        utils.print("Type 'q' to exit")

        while (query := input("\nInput : ").strip().lower()) != "q":
            await self.process_query(query)

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()


async def main(server_script_path):
    client = FiddlerMCPClient()
    server_script_path = os.path.abspath(server_script_path)
    try:
        await client.connect_to_server(server_script_path)
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        SERVER_PATH, help="Path to python script that acts as MCP server"
    )

    args = vars(parser.parse_args())
    server_script_path = args[SERVER_PATH]

    asyncio.run(main(server_script_path))
