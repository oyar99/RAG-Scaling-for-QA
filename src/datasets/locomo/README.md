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

The QA task can be evaluated using the `locomo_baseline_eval.py` script which uses an LLM to answer each question directly given the conversations as context.

## How to run

```sh
python .\locomo_baseline_eval.py -m gpt-4o-mini -c conv-26 -q 10 -ct 2
```

There are 5 different types of questions. Each question has a corresponding category from the below list.

- Multi-Hop (1): The model has to make multiple hops in the context to derive the correct answer.

- Temporal (2): The model has to answer the question with a date considering the date and time of the conversation.

- Open Domain (3): General broad questions about the conversation that require actual understanding.

- Single Hop (4): The model has to make a single observation from the conversation to answer the questions.

- Adversarial (5): The model has to choose between two types of answers, either a) Not mentioned or b) the actual answer.
