# 2WikiMultihopQA

[_2WikiMultihopQA_](https://github.com/Alab-NII/2wikimultihop) is an evaluation benchmark for Multi-Hop QA originally presented in the paper [Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps](https://aclanthology.org/2020.coling-main.580/). The dataset can be downloaded directly into this repository by running the following command:

```sh
.\download.sh
```

The corpus of the dataset can be extracted running the script under the `utils` directory.

```sh
python .\utils\generate_corpus.py
```

Please note this dataset has the same structure as HotpotQA.
