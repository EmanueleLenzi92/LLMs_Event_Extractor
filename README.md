# LLM Benchmark – Narrative-to-Events Segmentation

This script benchmarks multiple local LLMs (via **Ollama + LangChain**) for splitting **MOVING (https://www.moving-h2020.eu/) textual reports** into **events/paragraphs** without changing the original wording.

## What it does
- Runs several **models** × **system prompts** × **iterations** on a set of MOVING narratives
- Measures output quality by checking similarity to the original text (Jaccard) and basic paragraph/sentence statistics
- Saves all results to JSON files for analysis

## Inputs (configured in the script)
- `models` / `chosen_models`: LLMs to test (Ollama (https://ollama.com/) model names)
- `system_prompts` / `chosen_systemPrompt`: prompts for segmentation
- `iterations`: number of runs per model/prompt/text
- `chosen_narratives`: which MOVING texts to evaluate

## Outputs
The script writes JSON files (suffix is requested at runtime):
- `narratives_dict_output-<suffix>.json` → all raw generations per run
- `times_dict-<suffix>.json` → execution times (minutes)
- `final_dict-<suffix>.json` → best selected output per model/text
- `best_prompt-<suffix>.json` → best prompt chosen per model/text
- `metadata-<suffix>.json` → run configuration and narrative metadata

## Requirements
- Python 3.x
- Ollama running locally + the tested models installed
- Python packages: `langchain`, `nltk`

## Run
```bash
python main.py
```
You will be asked for a suffix to name the output JSON files.
