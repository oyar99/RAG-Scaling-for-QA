# HotpotQA

[_HotpotQA_](https://hotpotqa.github.io/) is an evaluation benchmark for QA featuring multi-hop questions. The dataset was constructed via _crowdsourcing_, where workers were tasked with creating questions that require context from multiple paragraphs, manually collected and curated from Wikipedia.

This dataset is included in this work primarily for completeness, as numerous studies have shown that many of its questions can be answered without truly effective multi-hop reasoning. However, including it allows us to evaluate how well our agent-based architectures perform on less challenging datasets.

Each sample contains a question, a context (10 paragraphs, including distractor passages), supporting facts, and the gold answer. The data is stored as JSON objects with the following format:

```json
{
  "_id": "5a8b57f25542995d1e6f1371",
  "answer": "yes",
  "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
  "supporting_facts": [
    ["Scott Derrickson", 0],
    ["Ed Wood", 0]
  ],
  "context": [
    [
      "Ed Wood (film)",
      [
        "Ed Wood is a 1994 American biographical ...",
        "The film concerns the period in Wood's ...",
        ...
      ]
    ],
    ...
  ],
  "type": "comparison",
  "level": "hard"
}
```

The dataset includes two types of questions:

1) __Multi-Hop question__: A question that, in theory, require multi-hop reasoning to answer.
(e.g., "What race track in the midwest hosts a 500 mile race eavery May?")

2) __Comparison question__: A question that compares two or more entities based on a specific feature.
(e.g., "Were Scott Derrickson and Ed Wood of the same nationality?")

Some of the files in this directory have been forked from the hotpot repo and adapted for QA and retrieval using only the dev set. See [_hotpot_](https://github.com/hotpotqa/hotpot)

To set up the dataset locally, please run the following command:

```sh
.\download.sh
```
