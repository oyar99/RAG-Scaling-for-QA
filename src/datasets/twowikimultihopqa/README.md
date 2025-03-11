# 2WikiMultihopQA

[_2WikiMultihopQA_](https://github.com/Alab-NII/2wikimultihop) is an evaluation benchmark for Multi-Hop QA originally presented in the paper [Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps](https://aclanthology.org/2020.coling-main.580/).

The dataset is constructed using hand-crafted templates with predefined logical rules, leveraging data from Wikipedia and Wikidata. Compared to __HotpotQA__, [_2WikiMultihopQA_](https://github.com/Alab-NII/2wikimultihop) includes evidence information for each sample in the form of triples (subject entity, property, object entity), designed to evaluate the reasoning and inference capabilities of language models. [_2WikiMultihopQA_](https://github.com/Alab-NII/2wikimultihop) is also more challenging than __HotpotQA__, as it aims to develop questions that do _require_ multi-hop reasoning. In contrast, studies have shown that many samples in __HotpotQA__ can be answered correctly without actual reasoning.

Each sample is stored as JSON object following a similar format as __HotpotQA__ as follows:

```json
{
  "_id": "8813f87c0bdd11eba7f7acde48001122",
  "type": "compositional",
  "question": "Who is the mother of the director of film Polish-Russian War (Film)?",
  "context": [
    [
      "Maheen Khan",
        [
          "Maheen Khan is a Pakistani fashion ...",
          "She has done many national and international ...",
          ...
        ]
    ],
    ...
  ],
  "supporting_facts": [
    [
      "Polish-Russian War (film)",
      1
    ],
    ...
  ],
  "evidences": [
    [
      "Polish-Russian War",
      "director",
      "Xawery \u017bu\u0142awski"
    ],
    ...
  ],
  "answer": "Ma\u0142gorzata Braunek"
}
```

There are 4 different types of questions.

1) Comparison question: A question that compares two or more entities with regards to a particular feature of that entity.
(e.g., <span style="color:blue">"Which film came out first, Blind Shaft or The Mask Of Fu Manchu?"</span>)

The dataset can be downloaded directly into this repository by running the following command:

```sh
.\download.sh
```

Please note this dataset has the same structure as HotpotQA.
