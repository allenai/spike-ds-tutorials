import string
import glob
import jsonlines
from random import sample, shuffle
import argparse


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
    if sentence_text in dev_and_test:
        return False
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


def collect_train_set_sentences(dataset_path):
    spike_matches_path = f"{args.datapath}/spike_matches"
    train_set = dict()
    dev_and_test = get_dev_and_test_sentences(dataset_path)
    invalids = 0
    same_sent = 0
    for file in glob.glob(f'{spike_matches_path}/**/*{args.suffix}.jsonl', recursive=True):
        with jsonlines.open(file, "r") as f:
            for sentence_dict in f:
                label = args.label if 'positive' in file else 'negative'
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
    print("Number of negatives: ", len([x for x, y in train_set.items() if y["label"] != 'positive']))
    print("Number of positives: ", len([x for x, y in train_set.items() if y["label"] == 'positive']))
    print("invalids: ", invalids)
    print("same_sent: ", same_sent)

    return train_set


# Alternatively, save as one token per line
def flatten_list(ent_list):
    return [item for sublist in ent_list for item in sublist]


def tag_sentence_one_token_per_row(sentence, with_person=True):
    if sentence["need_tagging"]:
        tags = []
        captures = [[*range(span[0], span[1] + 1)] for span in sentence["captures"]]
        entities = [[*range(span[0], span[1] + 1)] for span in sentence["entities"]]
        flat_captures = flatten_list(captures)
        flat_entities = flatten_list(entities)

        for i, word in enumerate(sentence['words']):
            if word != "'s":
                if i in flat_captures:
                    captures, tags = tag_span(captures, i, word, 'MUS', tags)
                elif i in flat_entities and with_person:
                    entities, tags = tag_span(entities, i, word, 'PER', tags)
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
        with open(fp.replace("train", "split_dev"), "w") as f:
            for line in all_lines[0:dev_border]:
                f.write(line)
        with open(fp.replace("train", "split_train"), "w") as f:
            for line in all_lines[dev_border:test_border]:
                f.write(line)
        with open(fp.replace("train", "split_test"), "w") as f:
            for line in all_lines[test_border:]:
                f.write(line)


def main():
    dataset_path = f"{args.datapath}/{args.dataset_name}"
    train_set = collect_train_set_sentences(dataset_path=dataset_path)

    with jsonlines.open(f'{dataset_path}/{args.version_name}.jsonl', 'w') as f:
        for sent in sample([v for v in train_set.values()], len(train_set)):
            tags = tag_sentence_one_token_per_row(sent, with_person=False)
            sent_json = {"id": sent["id"], "sent_items": tags}
            f.write(sent_json)
    split_train_dev_test(f'{dataset_path}/{args.version_name}.jsonl', sample=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--label', help='', default="positive")
    parser.add_argument('--datapath', help='', default="../data")
    parser.add_argument('--dataset_name', help='', default="musicians_dataset")
    parser.add_argument('--version_name', help='', default="all_without_person")
    parser.add_argument('--suffix', help='', default="_unique")

    args = parser.parse_args()
    main()
