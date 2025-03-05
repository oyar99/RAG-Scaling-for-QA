"""Dataset utilities for reading and processing datasets."""
import json
from typing import Callable, Any


def generate_corpus(
    input_path: str,
    output_path: str,
    context_extractor: Callable[[Any], dict],
    context_key='context'
) -> None:
    """Generates a corpus from the QA dataset.
    """
    with open(input_path, 'r', encoding='utf-8') as f, open(output_path, 'w', encoding='utf-8') as outfile:
        dataset = json.load(f)

        paragraphs = [
            context_extractor(context)
            for item in dataset
            for context in item[context_key]
        ]
        print(f'Generated {len(paragraphs)} paragraphs.')

        # Save the paragraphs to a new file
        json.dump(paragraphs, outfile, indent=4)
