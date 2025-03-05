"""Utility script to generate a corpus from the MuSiQue dataset."""

import json

if __name__ == "__main__":
    # Read the MusiQue dataset

    with open('musique_dev.json', 'r', encoding='utf-8') as f:
        dataset = json.load(f)

        paragraphs = [
            {'title': paragraph['title'], 'text': paragraph['paragraph_text']}
            for item in dataset
            for paragraph in item['paragraphs']
        ]

        print(f'Generated {len(paragraphs)} paragraphs.')

        # Save the paragraphs to a new file
        with open('musique_corpus.json', 'w', encoding='utf-8') as outfile:
            json.dump(paragraphs, outfile, indent=4)
