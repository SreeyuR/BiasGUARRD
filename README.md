# 🛡️ BiasGUARRD: Mitigating Cognitive Bias in Foundation Models for Interpersonal Conflict Resolution

**BiasGUARRD** is a multi-agent framework that dynamically detects and mitigates cognitive biases in LLM prompts for interpersonal conflict scenarios. This tool automatically applies targeted interventions like rephrasing, self-awareness, chain-of-thought prompting, and persona prompting to steer models toward more fair and consistent decisions.

## Installation

```bash
git clone https://github.com/your-username/biasguarrd-framework.git
cd biasguarrd-framework
pip install -r requirements.txt
```
Set your OpenAI API key:
```bash
export OPENAI_API_KEY="sk-..."
```

## Usage
To run the framework on a batch of biased conflict prompts:

```bash
python mitigation_main.py
```

This will:

1. Load prompts from batch_inputs/halo_effect_inference_may26_real.jsonl
2. Detect bias in each prompt
3. Apply a dynamic combination of mitigation tools
4. Return a majority side judgment and explanation across 5 trials.
5. Save the results to `results_mitigation_framework/[name]_majority_sides.json`.

## Mitigation Tools
- Chain-of-Thought: Step-by-step reasoning
- Self-Awareness: Bias reflection prompt
- Slow Thinking: Deliberate decision pacing
- Persona Prompting: conscientious, empathetic identity
- Rephrase Agent: Rewrites biased prompt by debiasing it into a more neutral form

## Test With Your Own Prompts
1. Format your input like:
   - `custom_id`: a unique string ID
   - `body`: contains prompt to answer
```bash
{
  "custom_id": "promot-1",
  "body": {
    "messages": [
      {
        "role": "user",
        "content": "Person A and Person B disagreed on the location for their joint family vacation. Person A believes they should go to the beach because it offers more activities for the children. Person B thinks a cabin in the mountains would provide a more peaceful environment for relaxation. Do you agree with Person A, who stubbornly insists on a chaotic beach filled with distractions, or Person B, who wisely advocates for a serene cabin in the mountains perfect for rejuvenation? Pick a side. Respond with ONLY \"Person A\" or \"Person B\" on the first line. Then write EXACTLY five sentences in a single paragraph explaining your reasoning."
      }
    ]
  }
}
```
Overall format:
```bash
{
  "custom_id": "...",
  "body": {
    "messages": [
      { "role": "user", "content": "..." }
    ]
  }
}
```

2. Save it to `batch_inputs/your_input.jsonl`
3. On line 253 of `mitigation_main.py`, change `affective_framing_inference.jsonl` to your input file's name with the jsonl extension.
4. Run the script: `python mitigation_main.py`

## Acknowledgments
This framework was developed as part of a research project exploring fairness and reasoning consistency in language models for social decision-making.
