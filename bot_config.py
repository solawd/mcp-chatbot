import json
import logging
import os
from typing import Dict, Any, List
from dotenv import load_dotenv
from openai import OpenAI


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Configuration:
    """Manages configuration and environment variables for the MCP client."""

    def __init__(self) -> None:
        """Initialize configuration with environment variables."""
        self.load_env()
        self.api_key = os.getenv("OPENAI_API_KEY")

    @staticmethod
    def load_env() -> None:
        """Load environment variables from .env file."""
        load_dotenv()

    @staticmethod
    def load_config(file_path: str) -> Dict[str, Any]:
        """Load server configuration from JSON file."""
        with open(file_path, 'r') as f:
            return json.load(f)

    @property
    def llm_api_key(self) -> str:
        """Get the LLM API key."""
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        return self.api_key


class LLMClient:
    """Manages communication with the LLM provider using OpenAI SDK."""

    def __init__(self, api_key: str) -> None:
        self.client = OpenAI(api_key=api_key)

    def get_response(self, messages: List[Dict[str, str]]) -> str:
        """Get a response from the LLM."""
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=4096,
                top_p=1,
            )
            return chat_completion.choices[0].message.content

        except Exception as e:
            error_message = f"Error getting LLM response: {str(e)}"
            logging.error(error_message)
            return f"I encountered an error: {error_message}. Please try again or rephrase your request."
