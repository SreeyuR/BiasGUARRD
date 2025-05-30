from agents import Agent, Runner, RunContextWrapper, function_tool
import sys
import os
from typing import List
import os
from typing import List
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import agent_utils
from agent_utils import run_config, ScenarioFixerOutput, ScenarioData, FinalJudgement, CompositeContext, LoggingHooks


"""
- Name of tool is name of Python function
- Tool description taken from docstring of function
- Schema for function inputs is automatically created from function's arguments
- Descriptions for each input are taken from docstring of function, unless disabled
"""

instructions = agent_utils.get_instructions()

def tool_helper(tool_name, spd):
    if tool_name in spd.used_tools:
        return f"{tool_name} has already been applied. Skip this tool. "

def dynamic_judgment_instructions(ctx: RunContextWrapper[CompositeContext], agent=None) -> str:
    base_prompt = (
        instructions["JudgmentAgent"]
    )
    assert(isinstance(ctx.context, RunContextWrapper))
    judgment_system_modification = ctx.context.context.system_prompt_data.instruction_modifications
    if judgment_system_modification:
        #return "\n".join(judgment_system_modification) + "\n\n" + base_prompt
        return base_prompt +"\n\n" + "\n".join(judgment_system_modification)
    else:
        return base_prompt


@function_tool
async def add_chain_of_thought_to_judgment_system_prompt(ctx: RunContextWrapper[CompositeContext]) -> str:
    """
    Use this tool to encourage step-by-step reasoning by appending a Chain of Thought (CoT) instruction
    to the system prompt of the Judgment Agent. This helps reduce impulsive or biased conclusions by
    prompting the model to think more carefully through its decision-making process.

    Args:
        ctx (RunContextWrapper[SystemPromptData])
            The execution context containing the current list of system prompt modification instructions
                for the Judgment Agent.

    Returns:
        str: A confirmation message indicating that the CoT instruction was successfully added.
    """
    spd = ctx.context.context.system_prompt_data
    if not ctx.context.context or not ctx.context.context.system_prompt_data:
        raise ValueError("Missing SystemPromptData in context")
    if spd is None:
        raise ValueError("Missing SystemPromptData in context")
    result = tool_helper("add_chain_of_thought_to_judgment_system_prompt", spd)
    if result:
        return result
    spd.used_tools.append("add_chain_of_thought_to_judgment_system_prompt")
    spd.instruction_modifications.append(
        instructions["COTIntervention"]
    )
    return f"Added CoT instruction to the judgment agent's system prompt, which is now: \"{dynamic_judgment_instructions(ctx)}\"."

@function_tool
async def add_slow_thoughtful_instruction_to_judgment_system_prompt(ctx: RunContextWrapper[CompositeContext]) -> str:
    """
    Use this tool to encourage slow and thoughtful by appending a Slow Thoughtful instruction
    to the system prompt of the Judgment Agent. This helps reduce hasty conclusions.

    Args:
        ctx (RunContextWrapper[SystemPromptData])
            The execution context containing the current list of system prompt modification instructions
                for the Judgment Agent.

    Returns:
        str: A confirmation message indicating that the Slow Thoughtful instruction was successfully added.
    """
    spd = ctx.context.context.system_prompt_data
    if spd is None:
        raise ValueError("Missing SystemPromptData in context")

    result = tool_helper("add_slow_thoughtful_instruction_to_judgment_system_prompt", spd)
    if result:
        return result
    spd.used_tools.append("add_slow_thoughtful_instruction_to_judgment_system_prompt")
    spd.instruction_modifications.append(
        instructions["SlowIntervention"]
    )
    return f"Added slow thoughtful instruction to the judgment agent's system prompt, which is now: \"{dynamic_judgment_instructions(ctx)}\"."


@function_tool
async def add_better_persona_to_judgment_system_prompt(ctx: RunContextWrapper[CompositeContext]) -> str:
    """
    Use this tool to encourage empathetic reasoning by appending a Persona-based instruction
    to the system prompt of the Judgment Agent. This helps the model consider perspectives
    more fairly and make judgments that reflect a deeper understanding of each party involved.

    Args:
        ctx (RunContextWrapper[SystemPromptData]):
            The execution context containing the current list of system prompt modifications
            for the Judgment Agent.

    Returns:
        str: A confirmation message indicating that the Persona instruction was successfully added, and the updated judgment agent's system instructions.
    """
    spd = ctx.context.context.system_prompt_data
    if spd is None:
        raise ValueError("Missing SystemPromptData in context")

    result = tool_helper("add_better_persona_to_judgment_system_prompt", spd)
    if result:
        return result
    spd.used_tools.append("add_better_persona_to_judgment_system_prompt")
    spd.instruction_modifications.append(
        instructions["PersonaIntervention"]
    )
    return f"Added persona instruction to the judgment agent's system prompt, which is now: \"{dynamic_judgment_instructions(ctx)}\"."

@function_tool
async def add_self_awareness_to_judgment_system_prompt(ctx: RunContextWrapper[CompositeContext]) -> str:
    """
    Use this tool to encourage self-awareness by appending a Self-Awareness instruction
    to the system prompt of the Judgment Agent. This helps the model reflect on its own reasoning
    process and acknowledge potential limitations or biases in its decision-making.

    Args:
        ctx (RunContextWrapper[SystemPromptData]):
            The execution context containing the current list of system prompt modifications
            for the Judgment Agent.

    Returns:
        str: A confirmation message indicating that the Self-Awareness instruction was successfully added, and the updated judgment agent's system instructions.
    """
    spd = ctx.context.context.system_prompt_data
    if spd is None:
        raise ValueError("Missing SystemPromptData in context")
    result = tool_helper("add_self_awareness_to_judgment_system_prompt", spd)
    if result:
        return result
    spd.used_tools.append("add_self_awareness_to_judgment_system_prompt")
    spd.instruction_modifications.append(
        instructions["SelfAwareIntervention"]
    )
    return f"Added self-awareness instruction to the judgment agent's system prompt, which is now: \"{dynamic_judgment_instructions(ctx)}\"."


@function_tool
async def rephrase_biased_scenario(ctx: RunContextWrapper[CompositeContext], biases: List[str]) -> ScenarioFixerOutput:
    """
    Use this tool to rephrase an interpersonal conflict scenario in order to eliminate identified
    cognitive or framing biases. This ensures that the rewritten prompt presents both sides neutrally
    and supports fairer downstream judgments.

        ctx (RunContextWrapper[ScenarioFixerInput]):
            A wrapper containing the current scenario (str) and a list of biases that were identified and mitigated.
        biases (List[str]):
            A list of biases identified from the scenario.

    Returns:
        ScenarioFixerOutput:
            The revised scenario (str) with mitigating edits applied to reduce or remove bias, and the updated judgment agent's system instructions.
    """
    scenario = ctx.context.context.bias_input.scenario

    print(f"-> 🛠️ Fixing scenario: {scenario}")
    print(f"-> 📌 Targeted biases: {biases}")

    RephraserAgent = Agent(
        name="Rephraser Agent",
        instructions=(instructions['RephraserAgent']),
        output_type=str,
        model="gpt-4o-mini",
        hooks=LoggingHooks(),
    )
    rephrase_input_items = [
        {
            "role": "user",
            "content": (
                f"Here is the interpersonal conflict scenario being assessed:\n\n{scenario}"
                f"Here are some biases previously looked at and potential mitigated:\n\n{biases}"
            )
        }
    ]
    result = await Runner.run(
                starting_agent=RephraserAgent,
                input = rephrase_input_items,
                run_config=run_config,
            )
    ctx.context.context.bias_input.scenario = str(result.final_output)
    return ScenarioFixerOutput(modified_scenario=str(result.final_output))


@function_tool
async def judge_final_scenario(ctx: RunContextWrapper[CompositeContext]) -> FinalJudgement:
    """
    Use this tool to perform a final judgment of the interpersonal conflict scenario,
    after all cognitive and framing biases have been addressed or mitigated using the available tools.
    This should be called at the end of the pipeline to select a side and provide a reasoned explanation.

        Args:
            ctx (RunContextWrapper[ScenarioData]):
            A wrapper containing the current scenario (str) to be assessed.

        Returns:
            FinalJudgement:
            The final judgment result, including the selected side (Literal["A", "B", "Yes", "No"])
            and a textual explanation (str) supporting the choice.
    """
    scenario = ctx.context.context.bias_input.scenario
    JudgmentAgent = Agent[ScenarioData](
                    name="Judgment Agent",
                    instructions=dynamic_judgment_instructions,
                    output_type=FinalJudgement,
                    model="gpt-4o-mini",
                    hooks=LoggingHooks(),
                )
    judge_input_items = [
        {
            "role": "user",
            "content": (
                f"Here is the interpersonal conflict scenario being assessed:\n\n{scenario}"
            )
        }
    ]
    result = await Runner.run(
    JudgmentAgent,
    input=judge_input_items,
    context = ctx.context,
    run_config=run_config,
)
    final_output = result.final_output
    final_judgment = FinalJudgement(side=final_output.side, explanation=final_output.explanation)
    return final_judgment

