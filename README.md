# Evaluating Cognitive Language Agent Architectures for Enabling Long-Term Memory in Large Language Models

_Cognitive Language Agents_ is a framework for designing intelling language agents that integrates LLMs for reasoning and communication, using language as their primary means of interaction with their environment. Language agents consist of three key components: memory, action, and decision-making.

In this work, we present a systematic evaluation of three agent-based architectures to implement long-term memory for Question Answering (QA) and Multi-Hop Question Answering (MHQA). We seek to evaluate how well these agent-based architectures perform in more general purpose tasks and how well their planning, collaboration and decision-making capabilites can be leveraged.

## Datasets

We evalute the 3 systems against well-known dataset benchmarks for QA and MHQA.

- _LoCoMo_: A dataset consisting of 10 _very_ long-term conversations between two users annotated for the QA task. The dataset has been forked into this repository under `datasets\locomo` folder. See [LoCoMo](https://github.com/snap-research/locomo) for more details on the generation and statistics of the dataset.

## Requirements

We recommend installing Python 3.9 and creating a virtual environment.

Verify isntalled version.

```bash
python --version
```

Create virtual environment.

```bash
python -m venv env
```

Activate the virtual environment.

```bash
.\env\Scripts\activate
```

Install required packages.

```bash
pip install -r requirements.txt
```
