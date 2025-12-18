import asyncio
import logging
import requests
from typing import Dict, List, Any


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MCPHttpServer:
    """Manages HTTP-based MCP server connections and tool execution."""

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        self.name: str = name
        self.config: Dict[str, Any] = config
        self.base_url: str = config.get('url', '')
        self.tools: List[Dict[str, Any]] = []

    async def initialize(self) -> None:
        """Initialize the HTTP server connection by fetching available tools."""
        try:
            # JSON-RPC 2.0 request for tools/list
            response = requests.post(
                self.base_url,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                    "params": {}
                },
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json"
                },
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            # Extract tools from JSON-RPC response
            if 'result' in data and 'tools' in data['result']:
                self.tools = data['result']['tools']
                logging.info(f"Server {self.name} initialized with {len(self.tools)} tools")
            else:
                logging.warning(f"No tools found in server {self.name}")

        except Exception as e:
            logging.error(f"Error initializing server {self.name}: {e}")
            raise

    def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the server."""
        return self.tools

    async def execute_tool(
            self,
            tool_name: str,
            arguments: Dict[str, Any],
            retries: int = 2,
            delay: float = 1.0
    ) -> Any:
        """Execute a tool with retry mechanism."""
        attempt = 0
        while attempt < retries:
            try:
                logging.info(f"Executing {tool_name}...")
                # JSON-RPC 2.0 request for tools/call
                response = requests.post(
                    self.base_url,
                    json={
                        "jsonrpc": "2.0",
                        "id": 2,
                        "method": "tools/call",
                        "params": {
                            "name": tool_name,
                            "arguments": arguments
                        }
                    },
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    },
                    timeout=30
                )
                response.raise_for_status()
                data = response.json()

                # Extract result from JSON-RPC response
                if 'result' in data:
                    return data['result']
                elif 'error' in data:
                    raise Exception(f"Tool execution error: {data['error']}")
                return data

            except Exception as e:
                attempt += 1
                logging.warning(f"Error executing tool: {e}. Attempt {attempt} of {retries}.")
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
                else:
                    logging.error("Max retries reached. Failing.")
                    raise
