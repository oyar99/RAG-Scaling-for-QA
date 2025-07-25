# Analyzing Retrieval Scaling in RAG Systems for Complex QA Benchmarks

In this work, we present a **systematic evaluation** of multiple RAG configurations using lexical (BM25) and semantic retrievers (msmarco-bert-base-dot-v,ColBERTv2), as well as graph-based approaches (HippoRAG).

## Datasets

We evaluate the various RAG systems against well-known benchmark datasets for **QA** and **MHQA (Multi-Hop Question Answering)**:

- **_LoCoMo_**: A dataset consisting of **10 very long-term conversations** between two users, annotated for the QA task. The dataset has been **forked into this repository** under the `src/datasets/locomo` directory. See [LoCoMo](https://github.com/snap-research/locomo) for details on dataset generation and statistics.

- **_HotpotQA_**: A QA dataset featuring **natural, multi-hop questions**. This dataset has been **forked into this repository** under the `src/datasets/hotpot` directory, with instructions on how to initialize it correctly.

- **_2WikiMultihopQA_**: A QA dataset to evaluate Multi-Hop questions that contains comprehensive information of reasoning paths required to arrive at the correct answer. This dataset can be found under `src/datasets/twowikimultihopqa` directory. See [2WikiMultihopQA](https://github.com/Alab-NII/2wikimultihop?tab=readme-ov-file) for more details on the dataset.

- **_MuSiQue_**: A Multi-Hop QA dataset with 2-4 hop questions constructeed via single-hop question composition. This dataset can be found under `src/datasets/musique` directory. See [MuSiQue](https://github.com/stonybrooknlp/musique) for more details on the dataset.

## Questions

Each dataset includes a subset of **different types of questions**, with a particular focus on **Multi-Hop questions**:

1. **Multi-Hop (1)**: The model must **perform multiple reasoning steps** across different parts of the conversation to derive the correct answer.
2. **Temporal (2)**: The model must answer a question that requires **understanding dates and times** within the conversation.
3. **Open-Domain (3)**: General broad questions about the conversation that require **deep comprehension**.
4. **Single-Hop (4)**: The model must **extract a single piece of information** from the conversation to answer the question.

## Requirements

The repo uses a wheel dependency that is supported only in Linux environments.

We recommend using **Python 3.10** and creating a virtual environment.

Verify the installed Python version:

```sh
python --version
```

Create virtual environment.

```bash
python -m venv env
```

Activate the virtual environment.

```bash
source env/bin/activate
```

Install required packages.

```bash
pip install -r requirements.txt
```

Define environment variables.

```env
AZURE_OPENAI_API_KEY= # Azure OpenAI key
AZURE_OPENAI_ENDPOINT= # Azure OpenAI endpoint
SCRIPT_LOG_LEVEL=DEBUG # Log level
CUDA_VISIBLE_DEVICES=2,3 # GPUS to use
LLM_ENDPOINT=http://localhost:8000/v1 # VLLM deployed model endpoint
REMOTE_LLM=1 # Whether the system should use VLLM deployed model or a cloud model
```

## Closed-Source Models

The script supports any **closed-source models** that allow **batch deployments** via the Azure Open AI API and **open-source models** that are available via VLLM.

Sample Models

| Model Name                  | Context Length   | Max Outputh Length |
|-----------------------------|------------------|--------------------|
| o3-mini                     | 200,000 tokens   | 100,000 tokens     |
| GPT-4o-mini                 | 128,000 tokens   | 16,384 tokens      |
| Qwen2.5-14B-Instruct        | 32,000 tokens    | 8,192 tokens       |
| Qwen2.5-1.5B-Instruct       | 32,768 tokens    | 8,192 tokens       |
| Gemma 3-27B                 | 128,000 tokens   | 8,192 tokens       |

## VLLM

To start a `VLLM` HTTP server, the following command can be used.

```sh
vllm serve Qwen/Qwen2.5-14B-Instruct --tensor-parallel-size 2 --dtype float16 --gpu-memory-utilization 0.95 --max-model-len 32000 --max-num-seqs 128
```

**Explanation:**

```sh
Qwen/Qwen2.5-14B-Instruct # Specifies the LLM model to use
--tensore-parallel-size # Specifies the number of GPUs to use using tensor parallelism
--dtype float16 # Loads the model using 16-bit floating point precision (fp16)
--gpu-memory-utilization # Specifies how much memory to use for each GPU
--max-model-len # Sets the maximum number of tokens per input sequence to 32,000
--max-nums-seqs # Maximum number of concurrent sequences (i.e., requests) that can be processed in parallel.
```

## How to run

The script supports two execution modes:

- `predict`: Generates answers for a given dataset.
- `eval`: Runs evaluation metrics (`Exact Match (EM)`, `R_1 Score`, `R_2 Score`, `L1 Score`) against ground-truth answers.

### Running Predictions

Below are examples of using the script in `predict` mode.

#### Example 1: Single-Hop Questions (LoCoMo Dataset)

To generate predictions for **up to 20 single-hop questions** from a single conversation in the _LoCoMo_ dataset using `gpt-4o-mini`, run:

```sh
python index.py -e predict -m gpt-4o-mini -c conv-26 -q 20 -ct 4 -d locomo -a bm25
```

**Explanation:**

```sh
-e predict    # Runs the script in prediction mode.
-m gpt-4o-mini    # Specifies GPT-4o-mini as the model.
-c conv-26    # Identifies the conversation ID to process. Change "conv-26" to target a different conversation or omit.
-q 20    # Limits the number of questions to at most 20.
-ct 4    # Filters only single-hop questions.
-d locomo    # Specifies the dataset (LoCoMo) to use.
-a bm25 # Specifies the RAG strategy (BM25) to use.
```

To choose the RAG system to use, the `-a` command line parameter can be used along with `-k` to indicate retrieval depth.

The QA results will be placed under `output/qa_jobs`, while retrieval results will be placed under `output/retrieval_jobs`.

#### Example 2: Multi-Hop Questions (HotpotQA Dataset)

To generate predictions for all multi-hop questions from up to 10 conversations in the _hotpotQA_ dataset using gpt-4o-mini, you can run the following command:

```sh
python index.py -e predict -m gpt-4o-mini -l 10 -ct 1 -d hotpot -a dense
```

**Explanation:**

```sh
-l 10    # Limits the number of conversations/samples to at most 10.
-ct 1    # Filters only multi-hop questions.
-d hotpot    # Specifies the dataset (hotpotQA) to use.
-a dense # Specifies the RAG strategy (msmarco-bert-base-dot-v) to use.
```

### Running Evaluation

To evalaute the generated predictions against ground truth using **Exact Match (EM)**, **R_1 Score**, and **R_2 Score**, run:

```sh
python index.py -e "eval" -ev "predictions.jsonl" -d hotpot
```

**Explanation:**

```sh
-e eval    # Runs the script in evaluation mode.
-ev predictions.jsonl    # Path to the batch output containing the generated answers.
```

The metrics will be logged in the app log file under `logs` with the respective id.

When the script is executed to compute **L1 Score**, LLM Judge results will be placed under `eval_jobs`.

### Getting Help

For more details on available command-line arguments, run:

```sh
python index.py --help
```
