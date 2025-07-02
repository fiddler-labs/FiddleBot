import os
import sys
import pdb
import json
import asyncio
import argparse
import streamlit as st

from uuid import uuid4
from rich import print
from dotenv import load_dotenv
from pydantic import BaseModel

# from openai import AsyncOpenAI
from langfuse.openai import openai, AsyncOpenAI
from typing import Optional

from opentelemetry import trace, baggage
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource, SERVICE_NAME
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    ConsoleSpanExporter,
)

import utils
import constants

import fdl_tracer
import fdl_queries
import fdl_plan_n_solve

load_dotenv()

LANGFUSE_PUBLIC_KEY = "LANGFUSE_PUBLIC_KEY"

OPENAI_API_KEY = "OPENAI_API_KEY"
GPT_41_MINI = "gpt-4.1"
MAX_TOKENS = 4096

USER_INPUT_PROMPT = "User: "
AI_RESPONSE_PROMPT = "AI: "
Q_EXIT_COMMAND = "q"
QUIT_EXIT_COMMAND = "quit"
EXIT_EXIT_COMMAND = "exit"
EXIT_COMMANDS = [Q_EXIT_COMMAND, QUIT_EXIT_COMMAND, EXIT_EXIT_COMMAND]
GOODBYE_MESSAGE = "Goodbye!"

ROLE = "role"
CONTENT = "content"

USER_ROLE = "user"
SYSTEM_ROLE = "system"
ASSISTANT_ROLE = "assistant"
TOOL_ROLE = "tool"

SERVER_PATH_ARG = "--server_path"
SERVER_PATH = "server_path"

FDL_SYSTEM_PROMPT = """
You are a helpful assistant that talks to users about Fiddler. Fiddler is the pioneer in AI Observability and Security, 
enabling organizations to build trustworthy and responsible AI systems. Our platform helps Data Science teams, 
MLOps engineers, and business stakeholders monitor, explain, analyze, and improve their AI deployments.

With Fiddler, you can:
- Monitor performance of ML models and generative AI applications
- Protect your LLM and GenAI applications with Guardrails
- Analyze model behavior to identify issues and opportunities
- Improve AI systems through actionable insights

The Fiddler platform is only available on the Preprod environment.

You will only respond to questions that are related to Fiddler. Ask the user to stick to Fiddler if they ask about other topics.
"""

TOOL_CALL_RESULT_TEMPLATE = (
    """The result of calling the tool {name} with arguments {args} is {result}"""
)

HAS_PERFORMANCE_QUERY_PROMPT = (
    "Does the user want to query performance metrics for a given model?"
)

# resource = Resource.create(attributes={SERVICE_NAME: constants.SERVICE_NAME})

# provider = TracerProvider(resource=resource)
# Sets the global default tracer provider
# trace.set_tracer_provider(provider)

# processor = SimpleSpanProcessor(OTLPSpanExporter(endpoint=constants.OTEL_ENDPOINT))
# provider.add_span_processor(processor)

# trace.get_tracer_provider().add_span_processor(processor)

# Creates a tracer from the global tracer provider
# tracer = trace.get_tracer("my.tracer.name")


class AsyncChatBot:
    def __init__(
        self,
        model_name: str = GPT_41_MINI,
        system_prompt: str = FDL_SYSTEM_PROMPT,
        server_script_path: str = None,
    ):
        """
        Initialize the AsyncChatBot with OpenAI API key.
        If no API key is provided, it will look for OPENAI_API_KEY in environment variables.
        """
        self.api_key = os.getenv(OPENAI_API_KEY)
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Please provide it or set OPENAI_API_KEY environment variable."
            )

        self.client = AsyncOpenAI(
            api_key=self.api_key,
        )
        self.model_name = model_name
        self.system_prompt = system_prompt
        system_message = utils.create_message(SYSTEM_ROLE, self.system_prompt)
        self.conversation_history = [system_message]

        # if not server_script_path:
        # raise ValueError("Server script path is required")

        self.plan_n_solve_client = fdl_plan_n_solve.FiddlerExecClient(
            server_script_path
        )

        otel_exp = os.getenv(constants.OTEL_EXP)

        self.fdl_tracer = fdl_tracer.FdlTracer.get_instance(
            tracer_name=constants.SERVICE_NAME, otel_export=otel_exp
        )
        self.tracer = self.fdl_tracer.get_tracer()

        self.available_tools = self.plan_n_solve_client.list_tools()

        self.session_id = str(uuid4())

    async def init_plan_n_solve(self):
        """Connect to the MCP server"""
        await self.plan_n_solve_client.connect_to_server()

    def get_user_input(self) -> str:
        """Get input from the user via CLI. Press 'q' to quit."""
        try:
            user_input = input(USER_INPUT_PROMPT).strip()
            with self.tracer.start_as_current_span(constants.USER_INPUT) as input_span:
                input_span.set_attribute(constants.USER_INPUT, user_input)

            return user_input
        except KeyboardInterrupt:
            utils.print(GOODBYE_MESSAGE)
            sys.exit(0)

    # async def has_performance_query(self, conversation_history: list) -> bool:
    #     """Check if the conversation history contains a performance query"""

    #     # user_message = conversation_history[-1]
    #     # modified_conversation_history = conversation_history[:-1]
    #     user_query = utils.create_message(USER_ROLE, HAS_PERFORMANCE_QUERY_PROMPT)
    #     modified_conversation_history = conversation_history + [user_query]

    #     class HasPerformanceQuery(BaseModel):
    #         reasoning: str
    #         has_performance_query: bool

    #     result = await self.client.responses.parse(
    #         model=self.model_name,
    #         input=modified_conversation_history,
    #         max_output_tokens=MAX_TOKENS,
    #         text_format=HasPerformanceQuery,
    #     )

    #     return result.output_parsed.has_performance_query

    # async def extract_project_and_model_name(
    #     self, conversation_history: list
    # ) -> tuple[str, str]:
    #     """Extract the project and model name from the conversation history"""
    #     user_query = utils.create_message(
    #         USER_ROLE, EXTRACT_PROJECT_AND_MODEL_NAME_PROMPT
    #     )
    #     modified_conversation_history = conversation_history + [user_query]

    #     return result.output_parsed.project_name, result.output_parsed.model_name

    async def get_llm_response(self, conversation_history: list) -> str:
        """Get response from OpenAI's GPT model asynchronously."""
        # Add user message to conversation history
        # user_message = utils.create_message(USER_ROLE, user_input)
        # self.conversation_history.append(user_message)

        # Get response from OpenAI asynchronously

        # if await self.has_performance_query(conversation_history):
        #     project_name, model_name = await self.extract_project_and_model_name(
        #         conversation_history
        #     )
        #     if project_name is None or model_name is None:
        #         # ask_user_for project and model name
        #         pass
        #     end_date = await self.extract_end_date(conversation_history)
        #     start_date = await self.extract_start_date(conversation_history)
        #     performance_df = fdl_queries.get_performance_metrics(
        #         project_name, model_name, start_date, end_date
        #     )

        with self.tracer.start_as_current_span(constants.LLM_RESPONSE) as llm_response:
            response = await self.client.chat.completions.create(
                model=self.model_name,
                messages=conversation_history,
                max_tokens=MAX_TOKENS,
                tools=self.available_tools,
                tool_choice="auto",
                session_id=self.session_id,
            )

            tool_call_results = []
            tool_call_ids = []

            if response.choices[0].message.tool_calls is not None:
                llm_response.set_attribute(
                    constants.TOOL_CHOICE, str(response.choices[0].message)
                )
                conversation_history.append(response.choices[0].message)

                for tool_call in response.choices[0].message.tool_calls:
                    with self.tracer.start_as_current_span(
                        constants.TOOL_CALL
                    ) as tool_call_span:
                        name = tool_call.function.name
                        args = json.loads(tool_call.function.arguments)
                        tool_id = tool_call.id

                        ## Propogate context to plan_n_solve_client
                        # ctx = baggage.set_baggage(constants.SESSION_ID, self.session_id)
                        ctx = {}
                        TraceContextTextMapPropagator().inject(ctx)

                        result = await self.plan_n_solve_client.plan_n_solve(
                            args, ctx, self.session_id
                        )
                        # tool_call_results.append(
                        # TOOL_CALL_RESULT_TEMPLATE.format(
                        # name=name, args=args, result=result
                        # )
                        # )
                        tool_call_results.append(result)
                        tool_call_ids.append(tool_id)

                        tool_call_span.set_attribute(constants.TOOL_CALL_ID, tool_id)
                        tool_call_span.set_attribute(constants.TOOL_CALL_NAME, name)
                        tool_call_span.set_attribute(
                            constants.TOOL_CALL_ARGS, json.dumps(args)
                        )
                        tool_call_span.set_attribute(
                            constants.TOOL_CALL_RESULTS, result
                        )

                for tool_call_result, tool_call_id in zip(
                    tool_call_results, tool_call_ids
                ):
                    tool_call_result_message = utils.create_message(
                        TOOL_ROLE, tool_call_result, tool_call_id=tool_call_id
                    )

                    conversation_history.append(tool_call_result_message)

                with self.tracer.start_as_current_span(
                    constants.LLM_TOOL_RESPONSE
                ) as llm_tool_response:
                    response = await self.client.chat.completions.create(
                        model=self.model_name,
                        messages=conversation_history,
                        max_tokens=MAX_TOKENS,
                        session_id=self.session_id,
                    )
                    ai_response = response.choices[0].message.content
                    llm_tool_response.set_attribute(constants.AI_RESPONSE, ai_response)
            else:
                # Extract and store the assistant's response
                ai_response = response.choices[0].message.content
                llm_response.set_attribute(constants.AI_RESPONSE, ai_response)

        assistant_message = utils.create_message(ASSISTANT_ROLE, ai_response)
        conversation_history.append(assistant_message)

        return conversation_history

    def get_system_message(self):
        """Get the system message for the chatbot"""
        system_message = utils.create_message(SYSTEM_ROLE, self.system_prompt)
        return system_message

    async def start_chat(self):
        """Start the async chat loop."""
        utils.print(
            f"Welcome to FiddleBot! Type {', '.join(EXIT_COMMANDS)} to end the conversation."
        )
        utils.print("Type your message and press Enter to chat with the AI.")
        system_message = self.get_system_message()
        conversation_history = [system_message]

        with self.tracer.start_as_current_span(constants.CHAT_LOOP) as chat_loop:
            chat_loop.set_attribute(constants.SYSTEM_PROMPT, self.system_prompt)
            while True:
                user_input = self.get_user_input()

                if user_input.lower() in EXIT_COMMANDS:
                    openai.flush_langfuse()
                    await self.plan_n_solve_client.cleanup()
                    utils.print(GOODBYE_MESSAGE)
                    sys.exit(0)

                user_message = utils.create_message(USER_ROLE, user_input)
                conversation_history.append(user_message)

                conversation_history = await self.get_llm_response(conversation_history)
                ai_response = conversation_history[-1][constants.CONTENT]

                # if await self.has_performance_query(conversation_history):
                #     project_name, model_name = (
                #         await self.extract_project_and_model_name(conversation_history)
                #     )
                #     if project_name is None or model_name is None:
                #         # ask_user_for project and model name
                #         pass
                #     end_date = await self.extract_end_date(conversation_history)
                #     start_date = await self.extract_start_date(conversation_history)
                #     performance_df = fdl_queries.get_performance_metrics(
                #         project_name, model_name, start_date, end_date
                #     )

                utils.print(f"{AI_RESPONSE_PROMPT}{ai_response}")


async def main():
    try:
        # Set up argument parser
        parser = argparse.ArgumentParser(description="FiddleBot Chat Interface")
        parser.add_argument(
            "-s", SERVER_PATH_ARG, type=str, help="Path to the MCP server script file"
        )
        args = vars(parser.parse_args())

        if args[SERVER_PATH] is not None:
            server_path = os.path.abspath(args[SERVER_PATH])
            if os.path.exists(server_path):
                chatbot = AsyncChatBot(server_script_path=server_path)
                await chatbot.init_plan_n_solve()
                await chatbot.start_chat()
        else:
            chatbot = AsyncChatBot()
            await chatbot.init_plan_n_solve()
            await chatbot.start_chat()

    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
