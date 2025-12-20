import asyncio
import json
import logging
import traceback
from typing import Dict, List, Any
import streamlit as st
import prompt_utils
from bot_config import LLMClient, Configuration
from mcp_util import MCPHttpServer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class ChatSession:
    """Orchestrates the interaction between user, LLM, and tools."""

    def __init__(self, servers: List[MCPHttpServer], llm_client: LLMClient) -> None:
        self.servers: List[MCPHttpServer] = servers
        self.llm_client: LLMClient = llm_client
        self.system_message: str = ""


    @staticmethod
    def format_tool_for_llm(tool: Dict[str, Any]) -> str:
        """Format tool information for LLM."""
        args_desc = []
        if 'inputSchema' in tool and 'properties' in tool['inputSchema']:
            for param_name, param_info in tool['inputSchema']['properties'].items():
                arg_desc = f"- {param_name}: {param_info.get('description', 'No description')}"
                if param_name in tool['inputSchema'].get('required', []):
                    arg_desc += " (required)"
                args_desc.append(arg_desc)

        return prompt_utils.get_tool_format_message(tool, args_desc)

    async def initialize(self) -> bool:
        """Initialize all servers and prepare system message."""
        # Initialize all servers
        for server in self.servers:
            try:
                await server.initialize()
            except Exception as e:
                logging.error(f"Failed to initialize server {server.name}: {e}")
                return False

        # Collect all tools
        all_tools = []
        for server in self.servers:
            tools = server.list_tools()
            all_tools.extend(tools)

        if not all_tools:
            logging.warning("No tools available from any server")
            return False

        tools_description = "\n".join([str(self.format_tool_for_llm(tool)) for tool in all_tools])

        self.system_message = prompt_utils.get_system_prompt(tools_description)

        return True

    async def process_llm_response(self, llm_response: str) -> str:
        """Process the LLM response and execute tools if needed."""
        try:
            tool_call = json.loads(llm_response)
            if "tool" in tool_call and "arguments" in tool_call:
                logging.info(f"Executing tool: {tool_call['tool']}")
                logging.info(f"With arguments: {tool_call['arguments']}")

                for server in self.servers:
                    tools = server.list_tools()
                    if any(tool['name'] == tool_call["tool"] for tool in tools):
                        try:
                            result = await server.execute_tool(tool_call["tool"], tool_call["arguments"])
                            return f"Tool execution result: {result}"
                        except Exception as e:
                            error_msg = f"Error executing tool: {str(e)}"
                            logging.error(error_msg)
                            return error_msg

                return f"No server found with tool: {tool_call['tool']}"
            return llm_response
        except json.JSONDecodeError:
            return llm_response

    async def get_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response for the user message."""
        # Add system message if not present
        if not messages or messages[0].get("role") != "system":
            messages.insert(0, {"role": "system", "content": self.system_message})

        llm_response = self.llm_client.get_response(messages)
        result = await self.process_llm_response(llm_response)

        if result != llm_response:
            # Tool was executed, get final response
            messages.append({"role": "assistant", "content": llm_response})
            messages.append({"role": "system", "content": result})

            final_response = self.llm_client.get_response(messages)
            return final_response
        else:
            # Direct response
            return llm_response


# Streamlit App
def main():
    st.set_page_config(page_title="MCP Chatbot", page_icon="🤖", layout="wide")

    st.title("🤖 MCP-Enabled Chatbot")
    st.markdown("Chat with an AI assistant that can access MCP tools for product and order management.")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "chat_session" not in st.session_state:
        st.session_state.chat_session = None

    if "initialized" not in st.session_state:
        st.session_state.initialized = False

    # Initialize chat session
    if not st.session_state.initialized:
        with st.spinner("Initializing MCP server connection..."):
            try:
                config = Configuration()
                server_config = config.load_config('servers_config.json')

                # Create servers based on configuration
                servers = []
                for name, srv_config in server_config['mcpServers'].items():
                    if srv_config.get('type') == 'streamable-http':
                        servers.append(MCPHttpServer(name, srv_config))
                    else:
                        logging.warning(f"Unsupported server type for {name}: {srv_config.get('type')}")

                if not servers:
                    st.error("No compatible servers found in configuration")
                    return

                llm_client = LLMClient(config.llm_api_key)
                chat_session = ChatSession(servers, llm_client)

                # Initialize servers
                success = asyncio.run(chat_session.initialize())

                if success:
                    st.session_state.chat_session = chat_session
                    st.session_state.initialized = True
                    st.success("✅ MCP server connected successfully!")
                else:
                    st.error("❌ Failed to initialize MCP server")
                    return

            except Exception as e:
                st.error(f"❌ Error initializing chatbot: {e}")
                logging.error(f"Initialization error: {traceback.print_exc()}")
                return

    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This chatbot uses the Model Context Protocol (MCP) to access:
        - Product catalog
        - Customer information
        - Order management

        Simply ask questions in natural language!
        """)

        st.header("📋 Available Tools")
        if st.session_state.chat_session:
            for server in st.session_state.chat_session.servers:
                tools = server.list_tools()
                for tool in tools:
                    with st.expander(f"🔧 {tool['name']}"):
                        st.markdown(f"**Description:** {tool.get('description', 'No description')}")

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask me anything..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Prepare messages for LLM
                    messages = [{"role": m["role"], "content": m["content"]}
                               for m in st.session_state.messages]

                    # Get response
                    response = asyncio.run(
                        st.session_state.chat_session.get_response(messages)
                    )

                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    logging.error(f"Chat error: {e}")


if __name__ == "__main__":
    main()
