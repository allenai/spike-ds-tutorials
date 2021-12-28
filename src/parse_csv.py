import jsonlines
import spacy
import argparse
import csv

nlp = spacy.load("en_core_web_sm")


def read_spike_output(fp):
    with open(fp, "r") as fin:
        reader = csv.DictReader(fin)
        if "positive" in reader.fieldnames:
            positive = True
            output_path = f"../data/spike_matches/positive/{args.prefix}exemplars.jsonl"
        else:
            positive = False
            output_path = f"../data/spike_matches/negative/{args.prefix}neg.jsonl"
        with jsonlines.open(output_path, "w") as fout:
            for row in reader:
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
                doc = nlp(row["sentence_text"])
                ent_span = dict()
                for token in doc:
                    sentence["words"].append(token.text)
                    if token.ent_type_ == args.superclass_tag:
                        try:
                            if (token.i in [0, len(doc)]) or (doc[token.i].ent_type_ != doc[token.i-1].ent_type_) and (doc[token.i].ent_type_ != doc[token.i+1].ent_type_):
                                ent_span.update({"first": token.i, "label": token.ent_type_, "last": token.i})
                                sentence["entities"].append(ent_span)
                                ent_span = dict()
                            elif token.i == 0 or (doc[token.i].ent_type_ != doc[token.i-1].ent_type_):
                                ent_span.update({"first": token.i, "label": token.ent_type_})
                            elif token.i == len(doc) or (doc[token.i].ent_type_ != doc[token.i+1].ent_type_):
                                ent_span.update({"last": token.i})
                                sentence["entities"].append(ent_span)
                                ent_span = dict()
                        except Exception as e:
                            print([t.text for t in doc], e) # TODO: fails when entity at end of sentence.

                fout.write(sentence)


def main():
    read_spike_output('../data/spike_csv/all_results.csv')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--superclass_tag', help='the type of entity you are looking for. If your desired capture is '
                                                 'not an entity, leave an empty string.', default="ORG")
    parser.add_argument('--prefix', help="If your'e making a version of the dataset and don't want to override the"
                                         " existing files.", default='')
    parser.add_argument('--max_duplicates', help='If True, each target entity will appear only once in the dataset.',
                        type=int, default=-1)
    parser.add_argument('--include_patterns', help="If True, sentences with patterns appear directly in the train set.",
                        dest="include_patterns", action="store_true")
    parser.add_argument('--add_negatives', help="If True, sentences with patterns appear directly in the train set.",
                        dest="add_negatives", action="store_true")
    args = parser.parse_args()
    main()
