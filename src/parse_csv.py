import jsonlines
import spacy
import argparse
import csv
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")


def get_tokens_info(row):
    entities = []
    words = []
    doc = nlp(row["sentence_text"])
    ent_span = dict()
    for token in doc:
        words.append(token.text)
        if args.superclass_tag:
            if token.ent_type_ == args.superclass_tag:
                try:
                    if (token.is_sent_start or token.is_sent_end) or (
                            doc[token.i].ent_type_ != doc[token.i - 1].ent_type_) and (
                            doc[token.i].ent_type_ != doc[token.i + 1].ent_type_):
                        ent_span.update({"first": token.i, "label": token.ent_type_, "last": token.i})
                        entities.append(ent_span)
                        ent_span = dict()
                    elif token.is_sent_start or (doc[token.i].ent_type_ != doc[token.i - 1].ent_type_):
                        ent_span.update({"first": token.i, "label": token.ent_type_})
                    elif token.is_sent_end or (doc[token.i].ent_type_ != doc[token.i + 1].ent_type_):
                        ent_span.update({"last": token.i})
                        entities.append(ent_span)
                        ent_span = dict()
                except Exception as e:
                    print([t.text for t in doc], e)
    return words, entities


def read_spike_output(fp):
    with open(fp, "r") as fin:
        reader = csv.DictReader(fin)
        if "positive" in reader.fieldnames:
            positive = True
            output_path = f"./data/spike_matches/positive/{args.prefix}_exemplars.jsonl"
        else:
            positive = False
            output_path = f"./data/spike_matches/negative/{args.prefix}_neg.jsonl"
        with jsonlines.open(output_path, "w") as fout:
            for row in tqdm(reader):
                idx = row["sentence_id"]
                if positive:
                    captures = {"positive": {
                        "first": row["positive_first_index"], "last": row["positive_last_index"]}}
                else:
                    captures = {}
                sentence = {
                    "words": [],
                    "captures": captures,
                    "sentence_index": idx,
                    "entities": []}
                words, entities = get_tokens_info(row)
                sentence.update({"words": words})
                if args.superclass_tag:
                    sentence.update({"entities": entities})
                fout.write(sentence)


def main():
    read_spike_output(args.csv_path)
    if args.negative_csv_path:
        read_spike_output(args.negative_csv_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', help="", default="")
    parser.add_argument('--negative_csv_path', help="", default="")
    parser.add_argument('--superclass_tag', help='the type of entity you are looking for. If your desired capture is '
                                                 'not an entity, leave an empty string.', default="")
    parser.add_argument('--prefix', help="If your'e making a version of the dataset and don't want to override the"
                                         " existing files.", default='')
    parser.add_argument('--max_duplicates', help='If True, each target entity will appear only once in the dataset.',
                        type=int, default=-1)
    args = parser.parse_args()
    main()
