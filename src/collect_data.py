import json
import requests
from tqdm import tqdm
from collections import defaultdict
import jsonlines
from random import shuffle
import argparse


def read_patterns_from_file(path):
    with open(path, "r") as f:
        return json.load(f)


def write_pattern_matches(pattern):
    pattern_matches = search_single_query(pattern)
    return pattern_matches


def search_single_query(pattern):
    spike_url = "https://spike.staging.apps.allenai.org"
    stream_location = get_stream_location(spike_url, pattern)
    if not stream_location: return None
    matches = list(search_stream(spike_url, stream_location, pattern))
    return matches


def get_lists(pattern):
    lists = defaultdict(list)
    list_names = pattern["lists"]
    if list_names:
        for name in list_names:
            with open(f"{args.list_path}/{name}.txt", "r") as f:
                for item in f.readlines():
                    lists[name].append(item.strip())
    return lists


def get_stream_location(spike_url, pattern):
    dataset = "wikipedia"
    url = spike_url + "/api/3/multi-search/query"
    query_type = pattern["type"]
    query = pattern["query"]
    case_strategy = pattern["case_strategy"]
    lists = get_lists(pattern)
    data = {
        "queries": {
            "main": {
                query_type: query
            }
        },
        "data_set_name": dataset,
        "context": {
            "lists": lists,
            "tables": {
            },
            "case_strategy": case_strategy,
            "attempt_fuzzy": False
        }
    }
    response = requests.post(url, json=data)
    if 'stream-location' not in response.headers:
        return None
    return response.headers['stream-location']


def search_stream(spike_url, stream_location, pattern):
    limit = pattern["limit"]
    stream_url = spike_url + stream_location + f"?include_sentence=true&limit={limit}&include_sentence=true"
    response = requests.get(stream_url)
    results = [json.loads(jl) for jl in response.text.splitlines()]
    if len(results) == 0:
        print(f"Couldn't find any results for pattern {pattern}")
    for result in results:
        try:
            if result.get('kind') in ['continuation_url', 'tip']: continue
            data = result['value']['sub_matches']['main']
            entities = [ent for ent in data['sentence']['entities'] if
                        ent["label"].startswith(args.superclass_tag)] if args.superclass_tag else []
            yield {
                'words': data['words'],
                'captures': data['captures'],
                'sentence_index': data['sentence_index'],
                'highlights': data['highlights'],
                'entities': entities
            }
        except Exception as e:
            raise Exception(result, e)


def get_capture_text(match, label):
    tokens = match["words"]
    first, last = match["captures"][label]["first"], match["captures"][label]["last"]
    return " ".join(tokens[first:last + 1])


def get_hearst_based_list_of_exemplars(patterns_dict: dict):
    exemplars = set()
    for idx, pattern in tqdm(patterns_dict.items()):
        matches = write_pattern_matches(pattern)
        for sent in matches:
            captures: dict = sent["captures"]
            first, last = captures["positive"]["first"], captures["positive"]["last"]
            if first != last:
                exemplar = " ".join(sent["words"][first:last+1])
                exemplars.add(exemplar)
    with open(f'{args.list_path}/exemplars.txt', "w") as f:
        for exemplar in exemplars:
            if exemplar:
                f.write(exemplar + "\n")
    print(f"created {len(exemplars)} exemplars.")
    return exemplars


def add_sentences_without_targets_or_subclass_to_dataset():
    meaningless_query = {
        "111": {
            "query": "the", "type": "boolean", "case_strategy": "ignore", "label": "negative", "lists": [],
            "limit": 4000
        },
        "222": {
            "query": "t=NN", "type": "boolean", "case_strategy": "ignore", "label": "negative", "lists": [],
            "limit": 4000
        },
        "333": {
            "query": f"e={args.distractor}", "type": "boolean", "case_strategy": "ignore", "label": "negative",
            "lists": [], "limit": 4000
        }
    }

    for idx, pattern in meaningless_query.items():
        with jsonlines.open(f'{args.spike_matches_dir}/negative/{idx}.jsonl', 'w') as f:
            matches = write_pattern_matches(pattern)
            shuffle(matches)
            print(f"number of matches for pattern {idx}: {len(matches)}")
            for match in matches:
                if not [x for x in match["entities"] if x["label"] == args.superclass_tag]:
                    f.write(match)

def main():
    patterns = read_patterns_from_file(f'./patterns/{args.patterns_file}')
    if args.hearst_patterns:
        hearst_patterns = read_patterns_from_file(f'./patterns/{args.hearst_patterns}')
        get_hearst_based_list_of_exemplars(hearst_patterns)
        patterns.update({
            "hearst": {
                "query": "positive:w={exemplars}",
                "type": "boolean",
                "case_strategy": "ignore",
                "label": "positive",
                "lists": ["exemplars"],
                "limit": int(args.hearst_limit)
            }
        })

        
def collect_matches_with_patterns(patterns):
    for idx, pattern in tqdm(patterns.items()):
        limit = pattern["limit"]
        try:
            max_sents = int(limit * 0.90)
        except:
            raise Exception(f"{idx}::: {pattern}::: {type(limit)}")
        label = pattern["label"]
        with jsonlines.open(f'{args.spike_matches_dir}/{label}/{args.prefix}{idx}.jsonl', 'w') as f:
            captures = defaultdict(lambda: 0)
            matches = write_pattern_matches(pattern)
            print(f"number of matches for pattern {idx}: {len(matches)}")
            shuffle(matches)
            if matches:
                print(f"1:::: len matches for pattern {idx}: {len(matches)}")
                matches = matches[:max_sents]
                print(f"2:::: len matches for pattern {idx}: {len(matches)}")
                for match in matches:
                    capture = get_capture_text(match, label)
                    if args.max_duplicates > 0:
                        if captures[capture] > args.max_duplicates:
                            continue
                    f.write(match)
                    captures[capture] += 1
                sorted_matches = {k: v for k, v in sorted(captures.items(), key=lambda item: item[1], reverse=True) if v > 1}
                if idx == "-1":
                    print(f"counter: {sorted_matches}")
            else:
                print("no matches")


def main():
    patterns = read_patterns_from_file(f'./patterns/{args.patterns_file}')
    if args.hearst_patterns:
        hearst_patterns = read_patterns_from_file(f'./patterns/{args.hearst_patterns}')
        get_hearst_based_list_of_exemplars(hearst_patterns)
        patterns.update({
            "-1": {
                "query": f"<E>positive:w={{exemplars}}&e={args.superclass_tag}",
                "type": "boolean",
                "case_strategy": "ignore",
                "label": "positive",
                "lists": ["exemplars"],
                "limit": int(args.hearst_limit)
            }
        })
    collect_matches_with_patterns(patterns)
    if args.distractor:
        add_sentences_without_targets_or_subclass_to_dataset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--superclass_tag', help='the type of entity you are looking for. If your desired capture is '
                                                 'not an entity, leave an empty string.', default="PERSON")
    parser.add_argument('--prefix', help="If your'e making a version of the dataset and don't want to override the"
                                         " existing files.", default='')
    parser.add_argument('--spike_matches_dir', help="", default='./data/spike_matches')
    parser.add_argument('--patterns_file', help="", default='hearst_patterns.json')
    parser.add_argument('--list_path', help="", default='./data/lists')
    parser.add_argument('--hearst_limit', help="max number of desired exemplars collected using hearst patterns",
                        default=30000)
    parser.add_argument('--hearst_patterns', help="file path to hearst patterns specific to your group. Leave empty "
                                                  "if you already have a list of exemplars. "
                                                  "If so, name your file 'exemplars.txt'", default='')
    parser.add_argument('--distractor', help="To enrich the dataset with sentences with no relevant entities at all, "
                                             "provide an entity type completely unrelated to your query. e.g. GPE",
                        default="")
    parser.add_argument('--max_duplicates', help='If True, each target entity will appear only once in the dataset.',
                        type=int, default=-1)
    args = parser.parse_args()
    main()

    # hearst_patterns = read_patterns_from_file(f'./patterns/{args.hearst_patterns}')
    # get_hearst_based_list_of_exemplars(hearst_patterns)
