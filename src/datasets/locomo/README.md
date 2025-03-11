# LoCoMo

[_LoCoMo_](https://github.com/snap-research/locomo) is an evaluation benchmark for very long-term conversational data consiting of 10 conversations properly annotated for the QA task.

The dataset has been modified to adapt it only for the QA evaluation benchmark.

Each conversation is stored as JSON object with the following format:

```json
{
  "qa": [
    {
      "question": "When did Melanie paint a sunrise?",
      "answer": 2022,
      "evidence": ["D1:12"],
      "category": 2,
    },
    {
      ...
    }
  ],
  "conversation": {
    "speaker_a": "Caroline",
    "speaker_b": "Melanie",
    "session_1_date_time": "1:56 pm on 8 May, 2023",
    "session_1": [
      {
        ...
      },
      {
        "speaker": "Caroline",
        "dia_id": "D1:13",
        "text": "Thanks, Melanie! That's really sweet. Is this your own painting?"
      },
      {
        "speaker": "Melanie",
        "dia_id": "D1:14",
        "text": "Yeah, I painted that lake sunrise last year! It's special to me."
      },
      {
        ...
      }
    ],
    "session_2_date_time": "1:14 pm on 25 May, 2023",
    "session_2": [
      ...
    ],
    ...
  },
  "sample_id": "conv-26"
}
```

Each conversation consists of multiple sessions that occurred at different times.

The QA task can be evaluated using the `index.py` with the appropriate command line arguments.

There are 5 different types of questions. Each question has a corresponding category from the below list.

1) Multi-Hop: A question that requires the model to do multi-hop reasoning to arrive at the correct answer.
(e.g., "Which city have both Jean and John visited?")

2) Temporal: A question that requires the model to reason about dates and times.
(e.g., "When did Maria donate her car?")

3) Open Domain: A broad question about the conversation that require an actual understanding of the context.
(e.g., "Would John be open to moving to another country?")

4) Single-Hop: A question that requires the model to do a single-hop reasoning step to answer.
(e.g., "What did Maria donate to a homeless shelter in December 2023?")

5) Adversarial: A question that may not be answerable because it's not explicitely or implicitely mentioned in the conversation.
(e.g., Q: "What country is Melanie's grandma from?" / A: "Not mentioned")
