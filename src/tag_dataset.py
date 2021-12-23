import string
import glob
import jsonlines
from random import sample, shuffle
import argparse
from pathlib import Path


def remove_tags(sentence):
    tokens = []
    for t in sentence.split():
        if t:
            tokens.append(t.split('-[',1)[0])
    return clean_punct(" ".join(tokens))


def clean_punct(sentence):
    s = sentence.translate(str.maketrans('', '', string.punctuation))
    s = s.replace("  ", " ")
    return s


def get_dev_and_test_sentences(dataset_path):
    test_path = dataset_path + '/test.txt'
    dev_path = dataset_path + '/dev.txt'
    with open(test_path, 'r') as ft, open(dev_path, 'r') as fd:
        test_set = [remove_tags(sent.strip()) for sent in ft.readlines()]
        dev_set = [remove_tags(sent.strip()) for sent in fd.readlines()]
    dev_and_test = dev_set + test_set
    return dev_and_test


def sentence_is_not_too_short(sentence_text):
    return len(sentence_text) > 50


def capture_is_not_non_alphabetical(capture_text):
    alphabet = 'q w e r t y u i o p a s d f g h j k l z x c v b n m'.split()
    return any(x in capture_text for x in alphabet)


def validate_sentence(capture_text, sentence_text, dev_and_test):
    if not capture_is_not_non_alphabetical(capture_text):
        return False
    if not sentence_is_not_too_short(sentence_text):
        return False
#     if sentence_text in dev_and_test:
#         return False
    return True


def get_capture(sentence, label):
    tokens = sentence["words"]
    capture = sentence['captures'].get(label)
    if capture:
        first = capture['first']
        last = capture['last']
        capture_tokens = [t for i, t in enumerate(tokens) if first <= i <= last]
        return " ".join(capture_tokens), first, last
    else:
        return "", -1, -1


def get_entities(sentence, cap_first, cap_last):
    entities = set()
    for e in sentence['entities']:
        all_entity_indices = [*range(e['first'], e['last'])]
        if all(x not in all_entity_indices for x in [cap_first, cap_last]):
            entities.add((e['first'], e['last']))
    return entities


def collect_train_set_sentences():
    spike_matches_path = f"./data/spike_matches"
    train_set = dict()
    dev_and_test = None  # get_dev_and_test_sentences(dataset_path)
    invalids = 0
    same_sent = 0
    for file in glob.glob(f'{spike_matches_path}/**/{args.prefix}*.jsonl', recursive=True):
        with jsonlines.open(file, "r") as f:
            if not args.include_patterns:
                if not (file.endswith("_neg.jsonl") or file.endswith("_exemplars.jsonl")):
                    print(file)
                    continue
            for sentence_dict in f:
                label = file.split("/")[-2]
                sentence_text = clean_punct(" ".join(sentence_dict["words"])).strip()
                capture_text, cap_first, cap_last = get_capture(sentence_dict, label)
                if capture_text:
                    if not validate_sentence(capture_text, sentence_text, dev_and_test):
                        invalids += 1
                        continue
                    if sentence_text not in train_set.keys():
                        if label == 'positive':
                            train_set[sentence_text] = {
                                "id": sentence_dict["sentence_index"],
                                "label": label,
                                "words": sentence_dict["words"],
                                "captures": {(cap_first, cap_last)},
                                "entities": get_entities(sentence_dict, cap_first, cap_last),
                                "need_tagging": True
                            }
                        else:
                            entities = get_entities(sentence_dict, cap_first, cap_last)
                            entities.update({(cap_first, cap_last)})
                            train_set[sentence_text] = {
                                "id": sentence_dict["sentence_index"],
                                "label": label,
                                "words": sentence_dict["words"],
                                "captures": {},
                                "entities": entities,
                                "need_tagging": True
                            }
                    else:
                        if label == 'positive':
                            train_set[sentence_text]["captures"].add((cap_first, cap_last))
                            new_entities = get_entities(sentence_dict, cap_first, cap_last)
                            train_set[sentence_text]["entities"].update(new_entities)
                        elif (cap_first, cap_last) not in train_set[sentence_text]["captures"]:
                            train_set[sentence_text]["entities"].add((cap_first, cap_last))
                        else:
                            # not a true negative!
                            continue
                else:
                    if sentence_text not in train_set:
                        train_set[sentence_text] = {
                            "id": sentence_dict["sentence_index"],
                            "label": label,
                            "words": sentence_dict["words"],
                            "captures": {},
                            "entities": {},
                            "need_tagging": False
                        }
                    else:
                        same_sent += 1

    # make sure there are significantly more negative examples than positive ones.
    negatives = [(x, y) for x, y in train_set.items() if y["label"] != 'positive']
    positives = [(x, y) for x, y in train_set.items() if y["label"] == 'positive']
    print("Number of negatives: ", len(negatives))
    print("Number of positives: ", len(positives))
    print("invalids: ", invalids)
    print("same_sent: ", same_sent)

    return train_set


# Alternatively, save as one token per line
def flatten_list(ent_list):
    return [item for sublist in ent_list for item in sublist]


def tag_sentence_one_token_per_row(sentence):
    if sentence["need_tagging"]:
        tags = []
        captures = [[*range(span[0], span[1] + 1)] for span in sentence["captures"]]
        entities = [[*range(span[0], span[1] + 1)] for span in sentence["entities"]]
        flat_captures = flatten_list(captures)
        flat_entities = flatten_list(entities)

        for i, word in enumerate(sentence['words']):
            if word != "'s":
                if i in flat_captures:
                    captures, tags = tag_span(captures, i, word, args.target_tag, tags)
                elif i in flat_entities and args.superclass_tag:
                    entities, tags = tag_span(entities, i, word, args.superclass_tag, tags)
                else:
                    tags.append((word, "O"))
            else:
                tags.append((word, "O"))
        return tags
    else:
        return [(word, "O") for word in sentence['words']]


def tag_span(span_list, i, word, tag_suffix, tags):
    for span in span_list:
        if i == span[0]:
            tags.append((word, f"B-{tag_suffix}"))
        elif i in span:
            tags.append((word, f"I-{tag_suffix}"))
        elif i == span[-1]:
            tags.append((word, f"I-{tag_suffix}"))
            span_list.remove(span)
    return span_list, tags


def split_train_dev_test(fp, sample=False):
    with open(fp, "r") as f:
        all_lines = f.readlines()
        shuffle(all_lines)
        datasize = len(all_lines)
        dev_border = int(datasize * 0.1) if not sample else 300
        test_border = int(datasize * 0.9) if not sample else datasize - 300
        with open(fp.replace(f"dataset", f"split_dev", 1), "w") as fdev:
            for line in all_lines[0:dev_border]:
                fdev.write(line)
        with open(fp.replace(f"dataset", f"split_train", 1), "w") as ftrain:
            for line in all_lines[dev_border:test_border]:
                ftrain.write(line)
        with open(fp.replace(f"dataset", f"split_test", 1), "w") as ftest:
            for line in all_lines[test_border:]:
                ftest.write(line)


def main():
    dataset_path = f"./data/{args.dataset}"
    Path(dataset_path).mkdir(parents=True, exist_ok=True)
    train_set = collect_train_set_sentences()

    with jsonlines.open(f'{dataset_path}/{args.prefix}dataset.jsonl', 'w') as f:
        for sent in sample([v for v in train_set.values()], len(train_set)):
            tags = tag_sentence_one_token_per_row(sent)
            sent_json = {"id": sent["id"], "sent_items": tags}
            f.write(sent_json)
    split_train_dev_test(f'{dataset_path}/{args.prefix}dataset.jsonl', sample=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', help='', default="schools")
    parser.add_argument('--prefix', help='', default="unique_")
    parser.add_argument('--target_tag', help='', default="MUS")
    parser.add_argument('--superclass_tag', help='', default="PER")
    parser.add_argument('--include_patterns', help="If True, sentences with patterns appear directly in the train set.",
                        dest="include_patterns", action="store_true")
    args = parser.parse_args()
    main()
