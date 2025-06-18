from rich import console, markdown

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
