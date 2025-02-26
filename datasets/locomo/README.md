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

The QA task can be evaluated using the `baseline-eval.py` script which uses an LLM to answer each question directly given the conversations as context.
