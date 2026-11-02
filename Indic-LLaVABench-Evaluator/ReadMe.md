
# LLaVA-Bench Evaluator
Image links: LLaVA-Bench: https://huggingface.co/datasets/MBZUAI/multilingual-llava-bench-in-the-wild/tree/main/images  
The question files for all Indian languages are stored in : `./inputs/`

## Supported Languages
The LLaVA-Bench Evaluator supports the following Indic languages:
- Hindi
- Telugu
- Tamil
- Kannada
- Malayalam
- Marathi
- Gujarati
- Bengali
- Sanskrit
- Assamese
- Odia
- English


## Usage Instructions

### Input
1. Place the model response file in the format specified in the example file in the following directory:
   ```
   ./model_response/responses_Telugu.jsonl
   ```
2. Follow the naming convention:
   ```
   responses_{language}.jsonl
   ```
   Replace `{language}` with the appropriate language identifier (e.g., Telugu, Hindi, etc.).

### Running the Scripts
To execute the evaluation script, use the following command in the terminal:
```bash
bash run.sh language
```
Replace `language` with the desired language (e.g., `Telugu`).

### Output
The output scores will be available in the following folder:
```
/scores/results.txt
```

## Notes
- Ensure that all required dependencies are installed before running the script.
- Check the example response file format to ensure your input file adheres to the expected structure.

## Acknowledgement

Indic-LLaVABench-Evaluator is built with reference to the code of the following projects: [LLaVA-1.5](https://github.com/haotian-liu/LLaVA/tree/main/llava/eval) and Dataset from: https://huggingface.co/datasets/MBZUAI/multilingual-llava-bench-in-the-wild/tree/main/images . Thanks for their awesome work!



## Judge model
`scripts/eval_gpt_review_bench.py` uses the `openai>=1.0` client and reads `OPENAI_API_KEY`. The judge defaults to `gpt-4-0613` (the original LLaVA-Bench setting). Override it with `--judge-model <name>` or the `JUDGE_MODEL` env var. Scores from different judges are not directly comparable. The response file must contain **one answer per question, in the order of `inputs/<language>/questions.jsonl`**: answers are paired with questions line by line, and the script refuses mismatched counts.

An end-to-end example (QLoRA fine-tuning of Qwen2-VL on Chitrakshara, then base-vs-fine-tuned scoring on Indic-LLaVA-Bench and Indic-POPE) is in [`../notebooks/Chitrakshara_Qwen2VL_QLoRA_BharatBench.ipynb`](../notebooks/Chitrakshara_Qwen2VL_QLoRA_BharatBench.ipynb).
