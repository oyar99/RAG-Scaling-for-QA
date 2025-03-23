# MuSiQue

[__MuSiQue_](https://github.com/stonybrooknlp/musique) is an evaluation benchmark for Multi-Hop QA, originally introduced in the paper [♫ MuSiQue: Multihop Questions via Single-hop Question Composition](https://aclanthology.org/2022.tacl-1.31/).

The dataset is constructed by composing single-hop questions from existing datasets in a way ensures they can only be answered through multi-hop reasoning, eliminating potential reasoning shortcuts. As a result, this dataset is even more challenging than both __HotpotQA__ and __2WikiMultiHopQA__, where models can sometimes reach the correct answer without using all supporting facts from the context. The resulting dataset consists of questions requiring 2-hop to 4-hop reasoning.

Each sample includes a question, the context (20 paragraphs), supporting facts, and the gold answer. The data is stored as JSON objects in the following format:

```json
{
  "id": "2hop__460946_294723",
  "paragraphs": [
    {
      "idx": 0,
      "title": "Grant's First Stand",
      "paragraph_text": "Grant's First Stand is the debut album by American jazz ...",
      "is_supporting": false
    },
    ...
    {
      "idx": 5,
      "title": "Miquette Giraudy",
      "paragraph_text": "Miquette Giraudy (born 9 February 1953, ...",
      "is_supporting": true
    },
  ],
  "question": "Who is the spouse of the Green performer?",
  "question_decomposition": [
        {
            "id": 460946,
            "question": "Green >> performer",
            "answer": "Steve Hillage",
            "paragraph_support_idx": 10
        },
        {
            "id": 294723,
            "question": "#1 >> spouse",
            "answer": "Miquette Giraudy",
            "paragraph_support_idx": 5
        }
  ],
  "answer": "Miquette Giraudy",
  "answer_aliases": [],
  "answerable": true
}
```

The dataset's questions follow six distinct reasoning graph structures, all of which can be broadly categorized as:

1) __Multi-Hop question__: A question that requires effective multi-hop reasoning, ranging from 2-hop to 4-hop inference.
(e.g., "Who was the sibling of Nannina de' Medici?")

The dataset can be downloaded directly into this repository by running:

```sh
.\download.sh
```
