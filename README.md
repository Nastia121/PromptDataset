# PromptDataset

This repository accompanies the bachelor thesis “Aging of Prompt Engineering Techniques Across Different AI Model Versions” and builds on the CodePromptEval Python dataset (Yu et al., 2024; Khojah et al., 2025). The original CodePromptEval benchmark consists of 7,072 prompts derived from 221 function-level code-generation tasks from CoderEval, where each task is instantiated with 32 unique combinations of five prompt engineering techniques: Few-shot learning, Persona, Chain-of-Thought, Function Signature (context), and List of Packages (context).

For this thesis, the dataset was cleaned and filtered to 218 function-level tasks, each evaluated under five Prompt Engineering Techniques: Zero-Shot, Few-Shot, Chain-of-Thought (CoT), Contrastive Chain-of-Thought (CCoT), and Program-of-Thought (PoT). With three independent generations per task–technique–model combination, this yields a total of 19,620 LLM runs across six instruction-tuned models: GPT-3.5-Turbo, GPT-4o, Qwen2 7B Instruct, Qwen2.5 7B Instruct, Mistral-7B-Instruct, and Mistral-Large.

The repository provides the modified dataset used in the experiments, the scripts for constructing prompts, running LLM inference, and extracting code, as well as the evaluation pipeline that reuses the original CoderEval Docker environments to compute pass@k metrics. It also includes the aggregated results that underpin the empirical analysis in the thesis.

## Repository structure

The main folders are organized as follows:

- `ccot_prompt_creation/`  
  Scripts and artifacts for constructing Contrastive Chain-of-Thought (CCoT) prompts.
  - `ccot.py`: Main script for generating CCoT prompts from base prompts.
  - `prompts.txt`: Source prompts used as input for CCoT construction.
  - `prompts_with_wrong_explanations.csv`: Generated prompts augmented with contrastive (wrong) explanations.

- `dataset/`  
  Datasets used in the experiments.
  - `my_dataset.csv`: Main cleaned dataset with tasks and prompts for most models.
  - `my_dataset_for_mistral_7b.csv`: Variant of the dataset adapted to Mistral-7B-specific prompt formatting.

- `format_converter/`  
  Utilities for converting between CSV and JSONL formats for evaluation.
  - `csv_to_jsonl_converter.py`: Converts CSV datasets to JSONL format.
  - `jsonl_to_csv_converter.py`: Converts JSONL predictions back to CSV.

- `llm_evaluation/`  
  Evaluation scripts that run tests on generated code using the CodePromptEval setup.
  - `run_tests.py`: Entry point for executing the test suites.

- `llm_generation/`  
  Code for calling LLMs via APIs and collecting their outputs.
  - `model_lib_openrouter.py`: Model wrapper for accessing LLMs through the OpenRouter API.
  - `model_lib_replicate.py`: Model wrapper for accessing LLMs through the Replicate API.
  - `run_llm.py`: Main script to run selected models on the dataset and store their responses.

- `results/`  
  Aggregated evaluation results.
  - `model_outputs/`: Raw cleaned LLM outputs before evaluation in jsonl.-format.
  - `failed_tasks.txt`: List of tasks where code generation failed.
  - `evaluation_results/`: Per-model evaluation outputs in jsonl.-format.
  
- `LICENSE`  
  License for this repository.

- `README.md`  
  Main documentation describing the project, dataset, and the structure of this repository.

- `requirements.txt`  
  Python package requirements for setting up the environment used in this repository.
