"""Utility script to generate a corpus from the 2WikimultihopQA dataset."""

import json

if __name__ == "__main__":
    # Read the 2wikimultihopQA dataset

    with open('dev.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)

        paragraphs = [
            {'title': context[0], 'text': ' '.join(context[1])}
            for item in dataset
            for context in item['context']
        ]

        print(f'Generated {len(paragraphs)} paragraphs.')

        # Save the paragraphs to a new file
        with open('corpus.json', 'w', encoding='utf-8') as outfile:
            json.dump(paragraphs, outfile, indent=4)
