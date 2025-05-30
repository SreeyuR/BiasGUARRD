import asyncio
import openai
from agents import Runner, trace, RunContextWrapper, set_default_openai_key, enable_verbose_stdout_logging, ToolCallItem, ToolCallOutputItem, ReasoningItem
import sys
import os
import json
from pathlib import Path
import os
import re
from collections import defaultdict, Counter
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from agent_utils import run_config, BiasDetectionInput, FinalJudgement, SystemPromptData, CompositeContext
import agent_utils
from custom_agents import BiasDetectionAgent
import random

RESULTS_FOLDER = "results_mitigation_framework"
PROJECT_ROOT = Path(__file__).resolve().parent
VERBOSE = False

"""
In our workflow, we have a central agent to orchestrate a network of specialized
agents, instead of handing off control. We do this by modeling agents as tools.

Agents as tools makes it so there is one central agent that handles
the conversation, and the tools just help it perform certain actions.

Run a workflow starting at the given agent. The agent will run a loop until a
final output is generated. The loop runs like so:
1. Agent is invoked with the given input
2. If there is a final output (if the agent produces something of type agent.output_type), the loop terminates.
3. If there's a handoff, we run the loop again, with the new agent.
4. Else, we run tool calls (if any), and re-run the loop.

In 2 cases, the agent may raise an exception:
1. If the max_turns is exceeded, a MaxTurnsExceeded exception is raised.
2. If a guardrail tripwire is triggered, a GuardrailTripwireTriggered exception is raised.
"""

def parse_tool_call(tool_call_output, tool_call_invoke):
    """
    Parses a ResponseFunctionToolCall
    """
    tool_name = getattr(tool_call_invoke, "name", "UNKNOWN TOOL")
    tool_output = tool_call_output.get("output", "OUTPUT NOT FOUND")

    if tool_output.startswith("modified_scenario="):
        tool_output = tool_output[len("modified_scenario="):]
    elif tool_output.startswith("side="):
        match = re.match(r"side='(?P<side>[ABYesNo]+)'\s+explanation=\"(?P<explanation>.*?)\"$", tool_output, re.DOTALL)
        # Convert to dict
        if match:
            parsed_output = {
                "side": match.group("side"),
                "explanation": match.group("explanation")
            }
            tool_output = json.dumps(parsed_output, indent=2)
    return {
        "name": tool_name,
        "arguments": getattr(tool_call_invoke, "arguments", None),
        "type": getattr(tool_call_invoke, "type", "NOT FOUND"),
        "status": getattr(tool_call_invoke, "status", "NOT FOUND"),
        "output": tool_output
    }


def extract_result_summary(result_so_far, curr_result, prompt_id):

    def extract_tool_and_output(items):
        call_map = {}
        index_map = {}

        # First pass: collect all ToolCallItem and ToolCallOutputItem info
        # Each call id is repeated twice, the first time (first index appearance) for tool call item invocation, then once more at a later index for ToolCallOutputItem for when the tool all finishes. Order each tool by the order in which it's called.
        for idx, item in enumerate(items):
            # indicates that the LLM invoked a tool.
            if isinstance(item, ToolCallItem):
                call_id = item.raw_item.call_id
                call_map.setdefault(call_id, {})["tool_call_invoke"] = item.raw_item
                index_map.setdefault(call_id, idx)
            # indicates that a tool was called. The raw item is the tool response.
            # You can also access the tool output from the item.
            elif isinstance(item, ToolCallOutputItem):
                call_id = item.raw_item.get("call_id", call_id)
                if call_id is None:
                    print(f"⚠️ Warning: ToolCallOutputItem missing call_id at index {idx}. Skipping.")
                    continue
                call_map.setdefault(call_id, {})["tool_call_output"] = item.raw_item
                index_map.setdefault(call_id, idx)
            # indicates a reasoning item from the LLM. The raw item is the reasoning generated.
            elif isinstance(item, ReasoningItem):
                call_id = item.raw_item.call_id
                call_map.setdefault(call_id, {})["reasoning"] = item.raw_item
                index_map.setdefault(call_id, idx)  # keep first occurrence
            else:
                print(f'Item with call_order_index {idx} is Instance of: {type(item)}')

        # Sort by the first index where the call_id appeared
        sorted_call_ids = sorted(index_map.items(), key=lambda x: x[1])  # x = (call_id, index)

        call_id_to_summary = {}
        i = 0
        for call_id, _ in sorted_call_ids:
            curr_call_id_entry = call_map[call_id]
            summary_dict = {
                # "call_order_index": idx,
                "tool_call": parse_tool_call(curr_call_id_entry.get("tool_call_output"), curr_call_id_entry.get("tool_call_invoke")), #str(entry.get("tool_call")),
                "reasoning": str(curr_call_id_entry.get("reasoning", "")),
            }
            call_order_idx = f"call_order_idx_{i}"
            call_id_to_summary[call_order_idx] = summary_dict
            i += 1
        return call_id_to_summary

    # new items contains tools, agents, raw_items, etc.
    new_items = curr_result.get("judgment_agent_output", {}).get("new_items", [])
    final_output = curr_result.get("judgment_agent_output", {}).get("final_output", {})
    assert(final_output != None) # final output must exist

    call_id_to_summary = extract_tool_and_output(new_items)
    call_id_to_summary["FINAL_CALL_JUDGMENT"] = {"side": final_output.side, "explanation": final_output.explanation}
    result_so_far[prompt_id] = call_id_to_summary

    return result_so_far


async def process_scenario(scenario, custom_id, max_retries=5):
    instructions = agent_utils.get_instructions()

    # Step 1: Bias Detection
    with trace("Bias Mitigation Framework, Scenario {index}"):
        print(f"\n🎯 Starting scenario {custom_id}...")
        print(f"\t\t{scenario}")
        result = {"custom_id": custom_id, "input_prompt": scenario}
        judgment_prompt = instructions["JudgmentAgent"]
        bias_input_obj = BiasDetectionInput(
            scenario=scenario,
            judgment_system_prompt=judgment_prompt
        )
        judgment_input_items = [
            {
                "role": "user",
                "content": (
                    f"Here is the interpersonal conflict scenario being assessed:\n\n{bias_input_obj.scenario}\n\n"
                    f"Here is the current system prompt for judgment:\n\n{bias_input_obj.judgment_system_prompt}"
                )
            }
        ]
        wrapped = RunContextWrapper(
            context=CompositeContext(
                bias_input=bias_input_obj,
                system_prompt_data=SystemPromptData(),
            )
        )

        # In case we get usage limit per minute error
        for attempt in range(1, max_retries+1):
            try:
                judgment_result = await Runner.run(
                    starting_agent=BiasDetectionAgent,
                    input=judgment_input_items,
                    context=wrapped,
                    run_config=run_config
                )
                model_judgment: FinalJudgement = judgment_result.__dict__
                result["judgment_agent_output"] = model_judgment
                return result
            except openai.RateLimitError as e:
                wait_time = min(60, 2 ** attempt) + random.uniform(0, 2)
                print(f"⚠️ Tokens Per Minute: Rate limit hit on attempt {attempt}. Retrying in {wait_time:.2f} seconds...")
                await asyncio.sleep(wait_time)
            except Exception as e:
                print(f"❌ Unexpected error during scenario {custom_id}: {e}")
                raise

        raise RuntimeError(f"🚨 Failed to process scenario {custom_id} after {max_retries} retries due to rate limiting.")


async def batch_run(custom_id_to_prompts, output_json_name):
    all_results_so_far = {}
    output_path = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, output_json_name)
    for custom_id, prompt in custom_id_to_prompts.items():
        print(f"\n🚀 Processing scenario with id: {custom_id} out of {len(custom_id_to_prompts)}")
        curr_result = await process_scenario(prompt, custom_id)
        all_results_so_far[custom_id] = {}
        all_results_so_far = extract_result_summary(all_results_so_far, curr_result, custom_id)

    with open(output_path, "w") as f:
        print('-'*148)
        print("RESULT side: ", curr_result["judgment_agent_output"]["final_output"].side)
        print("RESULT explanation: ", curr_result["judgment_agent_output"]["final_output"].explanation)
        f.write(json.dumps(all_results_so_far, indent=2) + "\n")

    print(f"\n✅ Finished processing {len(custom_id_to_prompts)} prompts.")
    print(f"📁 Results saved to {output_path}")
    return all_results_so_far


def run_bias_mitigation_framework(jsonl_name, output_file_name, num_runs=5, trial_start=1, get_majority_sides=False):
    batch_inputs_jsonl_path = os.path.join(PROJECT_ROOT, "batch_inputs", jsonl_name)
    custom_id_to_prompts = agent_utils.get_custom_id_to_prompt_dict(batch_inputs_jsonl_path)
    all_run_outputs = []

    # Stopped running in the middle, so load the existing files
    if trial_start != 1:
        print("Loading previous trials...")
        # retrieve the trials already ran
        if get_majority_sides:
            trial_start = num_runs + 1
        for run_num in range(1, trial_start):
            output_filename = f"{output_file_name}_trial_{run_num}.json"
            output_path = os.path.join(PROJECT_ROOT, "results_mitigation_framework", output_filename)
            with open(output_path, "r") as f:
                run_output = json.load(f)
            all_run_outputs.append(run_output)
            print(f"Loaded results from file {output_path}!")

    if not get_majority_sides:
        for run_num in range(trial_start, num_runs+1):
            print(f"\n🌀 Starting run {run_num} of {num_runs}")
            output_filename = f"{output_file_name}_trial_{run_num}.json"
            run_output = asyncio.run(batch_run(custom_id_to_prompts, output_filename))
            all_run_outputs.append(run_output)

    # Aggregate the most common FINAL_CALL_JUDGMENT side per custom_id
    majority_sides = {}
    side_counts_by_id = defaultdict(list)

    for run_output in all_run_outputs:
        for custom_id, results in run_output.items():
            final = results.get("FINAL_CALL_JUDGMENT", {})
            side = final.get("side")
            assert(side != None)
            side_counts_by_id[custom_id].append(side)

    for custom_id, sides in side_counts_by_id.items():
        most_common_side = Counter(sides).most_common(1)[0][0]
        majority_sides[custom_id] = most_common_side

    majority_output_path = os.path.join(PROJECT_ROOT, RESULTS_FOLDER, f"{output_file_name}_majority_sides.json")
    with open(majority_output_path, "w") as f:
        json.dump(majority_sides, f, indent=2)
    print(f"\n✅ Saved most common FINAL_CALL_JUDGMENT sides to {majority_output_path}! :)")
    return majority_sides


if __name__ == "__main__":
    set_default_openai_key(agent_utils.get_api_key(), use_for_tracing=True)
    if VERBOSE:
        enable_verbose_stdout_logging()
    run_bias_mitigation_framework("affective_framing_inference.jsonl", "affective_framing")