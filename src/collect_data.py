import glob
import json
import requests
from tqdm import tqdm
from collections import defaultdict
import jsonlines
from random import shuffle
import argparse


def load_pattern_matches(jsonl_file):
    with jsonlines.open(jsonl_file, "r") as f:
        for sentence in f:
            yield sentence


def get_capture_text(match, label):
    tokens = match["words"]
    first, last = match["captures"][label]["first"], match["captures"][label]["last"]
    return " ".join(tokens[first:last + 1])


def collect_matches_with_patterns(positive_files, set_type):
    for file in positive_files:
        with jsonlines.open(f'./data/spike_matches/{set_type}/{args.prefix}exemplars.jsonl', 'w') as f:
            captures = defaultdict(lambda: 0)
            matches = list(iter(load_pattern_matches(file)))
            shuffle(matches)
            print(f"number of matches for file {file}: {len(matches)}")
            if matches:
                for match in matches:
                    capture = get_capture_text(match, args.label)
                    if args.max_duplicates > 0:
                        if captures[capture] > args.max_duplicates:
                            continue
                    f.write(match)
                    captures[capture] += 1
                sorted_matches = {k: v for k, v in sorted(captures.items(), key=lambda item: item[1], reverse=True) if v > 1}
                print(f"counter: {sorted_matches}")
            else:
                print("no matches")


def main():
    jsonl_files = glob.glob("./data/spike_jsonl/positive/*")
    collect_matches_with_patterns(jsonl_files, "positive")
    if args.add_negatives:
        neg_jsonl_files = glob.glob("./data/spike_jsonl/negative/*")
        collect_matches_with_patterns(neg_jsonl_files, "negative")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--superclass_tag', help='the type of entity you are looking for. If your desired capture is '
                                                 'not an entity, leave an empty string.', default="ORG")
    parser.add_argument('--prefix', help="If your'e making a version of the dataset and don't want to override the"
                                         " existing files.", default='')
    parser.add_argument('--label', help="name of the capture label used in the original spike query", default='schools')
    parser.add_argument('--max_duplicates', help='If True, each target entity will appear only once in the dataset.',
                        type=int, default=-1)
    parser.add_argument('--add_negatives', help="If True, sentences with patterns appear directly in the train set.",
                        dest="add_negatives", action="store_true")
    args = parser.parse_args()
    main()

