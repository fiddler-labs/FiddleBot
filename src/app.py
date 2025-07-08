import sys
import yaml
import asyncio
import hashlib
import nest_asyncio
import streamlit as st
import streamlit_authenticator as stauth

import utils
import fdl_chat
import constants
import fdl_tracer
import fdl_queries

PROJECT_MODEL_NAME_ERROR_MESSAGE = "The project name and model name are required to query performance metrics. Please provide them."
START_END_DATE_ERROR_MESSAGE = "The start and end dates are required to query performance metrics. Please provide them."
PERFORMANCE_QUERY_ERROR_MESSAGE = (
    "Error in querying performance metrics. Please try again."
)
PERFORMANCE_QUERY_SUCCESS_MESSAGE = "Performance metrics for the model have been queried successfully. The graphs is displayed above."

st.set_page_config(page_title=constants.ST_TITLE, page_icon=constants.ST_FAVICON_PATH)

tasks = {}
fdl_queries = fdl_queries.FdlQueries()
# fdl_tracer = fdl_tracer.get_instance(tracer_name=constants.SERVICE_NAME, otel_export=os.getenv(constants.OTEL_EXP))
# tracer= fdl_tracer.get_tracer()

try:
    loop = asyncio.get_event_loop().set_debug(True)
except RuntimeError:
    loop = asyncio.new_event_loop()
    loop.set_debug(True)
    asyncio.set_event_loop(loop)


def schedule_task(key, coro):
    """Schedule an async task, mapped to a key"""
    if key not in tasks:
        tasks[key] = loop.create_task(coro)


def process_tasks():
    pending = []
    for task in tasks.values():
        if not task.done():
            pending.append(task)
    if pending:
        loop.run_until_complete(asyncio.gather(*pending))


def gen_key(*args):
    """Generate unique key"""
    return hashlib.sha256("-".join(map(str, args)).encode()).hexdigest()


if len(sys.argv) == 1:
    utils.print("Running with in-memory MCP server")
    mcp_server_path = None
else:
    mcp_server_path = sys.argv[1]


# @st.cache_resource
def load_chatbot(server_script_path):
    utils.print("Loading chatbot")
    fdl_chatbot = fdl_chat.AsyncChatBot(server_script_path=server_script_path)
    loop.run_until_complete(fdl_chatbot.init_plan_n_solve())
    return fdl_chatbot


fdl_chatbot = load_chatbot(mcp_server_path)

if constants.ST_CONVERSATION_HISTORY not in st.session_state:
    system_message = fdl_chatbot.get_system_message()
    conversation_history = [system_message]
    st.session_state[constants.ST_CONVERSATION_HISTORY] = conversation_history
else:
    conversation_history = st.session_state[constants.ST_CONVERSATION_HISTORY]


def handle_performance_query():
    """Handle the performance query"""
    utils.print("Handling performance query")

    ## conversatoin_history is from the global variable
    if fdl_queries.has_performance_query(conversation_history):
        utils.print("Has performance query")
        project_name, model_name = fdl_queries.extract_project_and_model_name(
            conversation_history
        )
        if project_name == "" or model_name == "":
            utils.print("Project name or model name is None")
            return PROJECT_MODEL_NAME_ERROR_MESSAGE

        start_date, end_date = fdl_queries.extract_start_end_date(conversation_history)
        if start_date is None or end_date is None:
            utils.print("Start date or end date is None")
            return START_END_DATE_ERROR_MESSAGE

        performance_df = fdl_queries.get_performance_metrics(
            project_name, model_name, start_date, end_date
        )
        if performance_df is None:
            utils.print("Performance df is None")
            return PERFORMANCE_QUERY_ERROR_MESSAGE
        else:
            for metric_name, metric_df in performance_df.items():
                st.markdown(f"## {metric_name}")
                st.line_chart(metric_df, x=constants.DATE, y=constants.VALUE)

            return PERFORMANCE_QUERY_SUCCESS_MESSAGE
    else:
        return False


# st.title(constants.ST_TITLE)

config = {
    constants.ST_CREDENTIALS: {
        constants.ST_USERNAMES: {
            constants.ST_ADMIN: {
                constants.ST_EMAIL: st.secrets[constants.ST_USER_EMAIL],
                constants.ST_NAME: st.secrets[constants.ST_USER_NAME],
                constants.ST_PASSWORD: st.secrets[constants.ST_USER_PASSWORD],
            }
        }
    },
}

authenticator = stauth.Authenticate(
    config[constants.ST_CREDENTIALS],
    st.secrets[constants.ST_COOKIE_NAME],
    st.secrets[constants.ST_COOKIE_KEY],
    st.secrets[constants.ST_COOKIE_EXPIRY_DAYS],
    auto_hash=False,  ## Setting to false as config.yaml contains hashed password
)

try:
    authenticator.login()
except Exception as e:
    st.error(f"Error: {e}")

auth_status = st.session_state.get(constants.ST_AUTH_STATUS)
if auth_status:
    st.header(constants.ST_TITLE)
    st.caption(constants.ST_FDL_ENV_MESSAGE)
    with st.sidebar:
        st.write(
            "FiddleBot is a chatbot that can help you with questions about projects and models, only on the Preprod environment."
        )
        st.write("FiddleBot has access the following capabilities via tools:")
        st.markdown(
            """
            - list all projects in fiddler
            - list all models in a project
            - get model schema
            - get model specs
            - list alert rules for a model
            - list triggered alerts for a rule
            - list all custom metrics for a model
            """
        )

    if constants.ST_MESSAGES not in st.session_state:
        st.session_state[constants.ST_MESSAGES] = []

    for message in st.session_state[constants.ST_MESSAGES]:
        st_role = message[constants.ST_ROLE]
        st_content = message[constants.ST_CONTENT]
        if st_role == constants.ST_FDL_ROLE:
            avatar = constants.ST_ICON_PATH
        else:
            avatar = None
        with st.chat_message(st_role, avatar=avatar):
            st.markdown(st_content)

    prompt = st.chat_input("What can FiddleBot help you with?")
    if prompt is not None:
        user_message = utils.create_message(constants.USER_ROLE, prompt)
        conversation_history.append(user_message)

        ## Show user message on screen
        with st.chat_message(constants.ST_USER_ROLE):
            # user_prompt = constants.ST_RIGHT_ALIGN.format(text=prompt)
            # st.markdown(user_prompt, unsafe_allow_html=True)
            st.markdown(prompt)

        ## Store user message in session state
        user_message = {
            constants.ST_ROLE: constants.ST_USER_ROLE,
            constants.ST_CONTENT: prompt,
        }
        st.session_state[constants.ST_MESSAGES].append(user_message)

        ## Echo response

        with st.spinner("Thinking..."):
            task_key = f"task-{gen_key(prompt)}"
            # if fdl_queries.has_performance_query(conversation_history):
            #     project_name, model_name = fdl_queries.extract_project_and_model_name(
            #         conversation_history
            #     )
            #     if project_name is None or model_name is None:
            #         # ask_user_for project and model name
            #         response = "The project name and model name are required to query performance metrics. Please provide them."
            #         ai_message = utils.create_message(
            #             constants.ASSISTANT_ROLE, response
            #         )
            #         conversation_history.append(ai_message)

            #     start_date, end_date = fdl_queries.extract_start_end_date(
            #         conversation_history
            #     )
            #     if start_date is None or end_date is None:
            #         response = "The start and end dates are required to query performance metrics. Please provide them."
            #         ai_message = utils.create_message(
            #             constants.ASSISTANT_ROLE, response
            #         )
            #         conversation_history.append(ai_message)

            #     performance_df = fdl_queries.get_performance_metrics(
            #         project_name, model_name, start_date, end_date
            #     )
            #     if performance_df is None:
            #         response = (
            #             "Error in querying performance metrics. Please try again."
            #         )
            #         ai_message = utils.create_message(
            #             constants.ASSISTANT_ROLE, response
            #         )
            #         conversation_history.append(ai_message)
            #     else:
            #         for metric_name, metric_df in performance_df.items():
            #             st.markdown(f"## {metric_name}")
            #             st.line_chart(metric_df)

            #         response = "Performance metrics for the model have been queried successfully. The graphs is displayed above."
            #         ai_message = utils.create_message(
            #             constants.ASSISTANT_ROLE, response
            #         )
            #         conversation_history.append(ai_message)

            performance_query_response = handle_performance_query()
            if isinstance(performance_query_response, str):
                response = performance_query_response
                ai_message = utils.create_message(constants.ASSISTANT_ROLE, response)
                conversation_history.append(ai_message)
                with st.chat_message(
                    constants.ST_FDL_ROLE, avatar=constants.ST_ICON_PATH
                ):
                    st.markdown(response)
            else:
                schedule_task(
                    task_key, fdl_chatbot.get_llm_response(conversation_history)
                )
                process_tasks()

                if task_key in tasks:
                    task = tasks[task_key]

                    if task.done():
                        ## Path is relative to the directory from where the app is run
                        conversation_history = task.result()
                        response = conversation_history[-1][constants.CONTENT]
                        with st.chat_message(
                            constants.ST_FDL_ROLE, avatar=constants.ST_ICON_PATH
                        ):
                            st.markdown(response)

        ## Store response in session state
        fdl_message = {
            constants.ST_ROLE: constants.ST_FDL_ROLE,
            constants.ST_CONTENT: response,
        }
        st.session_state[constants.ST_MESSAGES].append(fdl_message)
        st.session_state[constants.ST_CONVERSATION_HISTORY] = conversation_history
elif auth_status is False:
    st.error("Invalid Credentials")
else:
    st.warning("Please enter your credentials")
