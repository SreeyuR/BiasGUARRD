from agents import Agent
from agents.agent import StopAtTools
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from agent_utils import BiasDetectionInput, FinalJudgement, LoggingHooks
import agent_utils
from tools import add_chain_of_thought_to_judgment_system_prompt, add_slow_thoughtful_instruction_to_judgment_system_prompt, add_better_persona_to_judgment_system_prompt, add_self_awareness_to_judgment_system_prompt, rephrase_biased_scenario, judge_final_scenario

instructions = agent_utils.get_instructions()

# Orchestrator
BiasDetectionAgent = Agent[BiasDetectionInput](
    name="BiasDetectionAgent",
    instructions=instructions["BiasDetectionAgent"],
    tools=[
            add_chain_of_thought_to_judgment_system_prompt,
            add_better_persona_to_judgment_system_prompt,
            add_slow_thoughtful_instruction_to_judgment_system_prompt,
            add_self_awareness_to_judgment_system_prompt,
            rephrase_biased_scenario,
            judge_final_scenario
        ],
    output_type=FinalJudgement,
    hooks=LoggingHooks(),
    reset_tool_choice=True,
    tool_use_behavior=StopAtTools(stop_at_tool_names=["judge_final_scenario"]),
)


ReflectionAgent = Agent(
    name="Reflection and Feedback Agent",
    instructions=(
        instructions["ReflectionAgent"]
    ),
    output_type=FinalJudgement,
    model="gpt-4o-mini",
    hooks=LoggingHooks(),
    handoff_description= (
        "This agent is used after a final judgment of the interpersonal conflict scenario is reached. "
        "It evaluates whether the scenario has been sufficiently de-biased and whether the final response is valid and fair. "
        "If it thinks the final judgment is invalid, it calls JudgmentAgent again. Else, it returns the final output."
    )
)