"""Pretty print JSON lines in a file."""

import json

INPUT_FILE = 'data/musique_ans_v1.0_dev.jsonl'
OUTPUT_FILE = 'musique_dev.json'


def prettify_json_lines(input_file: str, output_file: str) -> None:
    """Pretty print JSON lines in a file.

    Args:
        input_file (str): the input jsonl file path
        output_file (str): the output jsonl file path
    """
    entries = []

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            json_obj = json.loads(line)
            entries.append(json_obj)

        pretty_json = json.dumps(entries, indent=4)
        outfile.write(pretty_json + '\n')


if __name__ == "__main__":
    prettify_json_lines(INPUT_FILE, OUTPUT_FILE)
