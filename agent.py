from google.genai import types
from system_prompts import (
    EXTRACT_TEXT_SYSTEM_INSTRUCTION,
    FIND_COORDINATES_SYSTEM_INSTRUCTION,
    NEXT_ACTION_COMPUTER_USE_SUFFIX,
    NEXT_ACTION_JSON_SCHEMA,
    NEXT_ACTION_SYSTEM_INSTRUCTION,
    TABLE_ROWS_COMPUTER_USE_SUFFIX,
    TABLE_ROWS_SYSTEM_INSTRUCTION,
)


class AgentConfig:
    def __init__(
        self,
        model: str,
        excluded_functions: list[str],
        function_declarations: list[types.FunctionDeclaration],
        w0: int,
        h0: int,
        choice: str,
    ):
        self.model = model
        self.excluded_functions = excluded_functions
        self.function_declarations = function_declarations
        self.w0 = 1000
        self.h0 = 1000
        self.choice = choice

    def system_prompt(self, choice: str) -> str:
        if choice == "1":
            return FIND_COORDINATES_SYSTEM_INSTRUCTION
        if choice == "2":
            return EXTRACT_TEXT_SYSTEM_INSTRUCTION
        if choice == "3":
            return NEXT_ACTION_SYSTEM_INSTRUCTION
        if choice == "4":
            return TABLE_ROWS_SYSTEM_INSTRUCTION
        raise ValueError(f"Invalid choice: {choice}")

    def get_config(self) -> types.GenerateContentConfig:
        instruction = self.system_prompt(self.choice)
        if "computer-use" in self.model.lower():
            if self.choice == "3":
                instruction += NEXT_ACTION_COMPUTER_USE_SUFFIX
            if self.choice == "4":
                instruction += TABLE_ROWS_COMPUTER_USE_SUFFIX
            cfg: dict = {
                "system_instruction": instruction,
                "tools": [
                    types.Tool(
                        computer_use=types.ComputerUse(
                            excluded_predefined_functions=self.excluded_functions,
                        )
                    ),
                ],
            }
            if self.choice == "4":
                cfg["temperature"] = 0.1
                cfg["max_output_tokens"] = 8192
                cfg["tool_config"] = types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode=types.FunctionCallingConfigMode.NONE,
                    ),
                )
            return types.GenerateContentConfig(**cfg)
        if self.choice == "3":
            return types.GenerateContentConfig(
                system_instruction=instruction
                + "\n\nRespond with JSON only; match the API response schema (no markdown fences).",
                temperature=0.2,
                max_output_tokens=2048,
                tools=[
                    types.Tool(function_declarations=self.function_declarations)
                ],
                response_mime_type="application/json",
                response_json_schema=NEXT_ACTION_JSON_SCHEMA,
            )
        return types.GenerateContentConfig(
            system_instruction=instruction,
            tools=[types.Tool(function_declarations=self.function_declarations)],
        )
