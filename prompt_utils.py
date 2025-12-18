# Utility functions for generating system and tool prompts

def get_system_prompt(tools_description):
    system_message = f"""You are a helpful assistant with access to these tools:
{tools_description}
Choose the appropriate tool based on the user's question. If no tool is needed, reply directly.

IMPORTANT: When you need to use a tool, you must ONLY respond with the exact JSON object format below, nothing else:
{{
    "tool": "tool-name",
    "arguments": {{
        "argument-name": "value"
    }}
}}

After receiving a tool's response:
1. Transform the raw data into a natural, conversational response
2. Keep responses concise but informative
3. Focus on the most relevant information
4. Use appropriate context from the user's question
5. Avoid simply repeating the raw data

Please use only the tools that are explicitly defined above."""
    return system_message

def get_tool_format_message(tool, args_desc):
    f"""Tool: {tool['name']}
Description: {tool.get('description', 'No description')}
Arguments:
{chr(10).join(args_desc) if args_desc else 'No arguments'}"""
