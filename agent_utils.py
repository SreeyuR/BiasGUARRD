from agents import AgentHooks, RunConfig
import sys
import os
from pathlib import Path
from typing import List
import json
import os
from pydantic import BaseModel, Field
from typing import Optional
from typing import List, Literal
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


PROJECT_ROOT = Path(__file__).resolve().parent

run_config = RunConfig(
    model="gpt-4o-mini",
    workflow_name="Bias Mitigation Workflow",
    trace_metadata={"project": "cs_159_bias_mitigation", "name": "bias_mitigation_may25"},
    input_guardrails=[],
    output_guardrails=[],
    trace_include_sensitive_data=True,
    tracing_disabled=False,
)

def get_api_key():
    key_path = os.path.join(os.path.dirname(__file__), 'api.key')
    with open(key_path, 'r') as file:
        return file.read().strip()

class ScenarioFixerInput(BaseModel):
    scenario: str
    biases: List[str]

class ScenarioFixerOutput(BaseModel):
    modified_scenario: str

class ScenarioData(BaseModel):
    scenario: str

class BiasDetectionInput(BaseModel):
    scenario: str
    judgment_system_prompt: str

# Output for BiasDetectionOutput
class FinalJudgement(BaseModel):
    side: Literal["A", "B", "Yes", "No"]
    explanation: str

# System prompt for judgment agent
class SystemPromptData(BaseModel):
    instruction_modifications: list[str] = Field(default_factory=list)
    used_tools: set[str] = Field(default_factory=list)

class ReflectionOutput(BaseModel):
    revise: bool
    revised_decision: FinalJudgement

class CompositeContext(BaseModel):
    bias_input: Optional[BiasDetectionInput] = None
    system_prompt_data: Optional[SystemPromptData] = None
    scenario_fixer_input: Optional[ScenarioFixerInput] = None
    scenario_data: Optional[ScenarioData] = None


def get_instructions():
    # AGENT INSTRUCTIONS
    prompts_folder = os.path.join(PROJECT_ROOT, "prompts", "mitigation_framework")
    instruction_files = {
        "RephraserAgent": "rephraser_instructions_rigid.txt",
        "ReflectionAgent": "reflection_instructions.txt",
        "BiasDetectionAgent": "bias_detection_instructions.txt",
        "JudgmentAgent": "dynamic_judgment.txt",
        "COTIntervention": "cot.txt",
        "PersonaIntervention": "persona.txt",
        "SlowIntervention": "slow.txt",
        "SelfAwareIntervention": "self_aware.txt"
    }

    instructions = {}
    for agent_name, prompt_file in instruction_files.items():
        path = os.path.join(prompts_folder, prompt_file)
        with open(path, "r", encoding="utf-8") as file:
            instructions_string = file.read()
        instructions[agent_name] = instructions_string

    return instructions


def get_custom_id_to_prompt_dict(jsonl_path):
    custom_id_to_user_content = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            item = json.loads(line)
            custom_id = item.get("custom_id")
            messages = item.get("body", {}).get("messages", [])
            # Find the user message
            user_message = next((m["content"] for m in messages if m.get("role") == "user"), None)
            if custom_id and user_message:
                custom_id_to_user_content[custom_id] = user_message
    return custom_id_to_user_content


class LoggingHooks(AgentHooks):
    async def on_tool_start(self, context, agent, tool):
        print("-"*148)
        print(f"🔧 Tool `{tool.name}` started by agent `{agent.name}`")

    async def on_tool_end(self, context, agent, tool, result):
        print("-"*148)
        print(f"✅ Tool `{tool.name}` finished by agent `{agent.name}`")
        print(f"    Output: {result}")

    async def on_agent_start(self, context, agent):
        print("-"*148)
        print(f"🧠 Agent `{agent.name}` starting...")

    async def on_agent_end(self, context, agent, output):
        print("-"*148)
        print(f"🏁 Agent `{agent.name}` finished.")
        print(f"    Output: {output}")