import logging

from simpletransformers.ner import NERModel
import re
import json
import jsonlines
import argparse
import pandas as pd
import numpy as np
import shutil

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def set_args():
    return {
            "seed": 42,
            "labels_list": LABELS,
            "reprocess_input_data": True,
            "overwrite_output_dir": True,
            "train_batch_size": BATCH_SIZE,
            "eval_batch_size": BATCH_SIZE,
            "num_train_epochs": EPOCHS,
            "save_eval_checkpoints": False,
            "save_steps": -1,
            "use_multiprocessing": False,
            "use_multiprocessing_for_evaluation": False,
            "evaluate_during_training": True,
            "evaluate_during_training_steps": 50,
            "evaluate_during_training_verbose": True,
            "fp16": False,
            "wandb_project": args.wandb_project,
            "learning_rate": 0.0003,
            "warmup_ratio": 0.1,
            "logging_steps": 1,
            "best_model_dir": best_model_dir
        }


# def ingest_data_to_df(filepath, label_map):
#     tagged_data = []
#     sentence_number = 0
#     with open(filepath, 'r') as f:
#         for line in f.readlines():
#             for tagged_word in line.split():
#                 try:
#                     word, tag = tagged_word.split("-[", 1)
#                     old_tag = tag.split("]", 1)[0]
#                     new_tag = label_map[old_tag]
#                     tagged_data.append([sentence_number, word, new_tag])
#                 except Exception as e:
#                     raise Exception((tagged_word, line), e)
#             sentence_number += 1
#     return pd.DataFrame(tagged_data, columns=["sentence_id", "words", "labels"])


# def untag_dev_sentences(devpath):
#     output_path = f"sentences_{devpath}"
#     clean_sentences = []
#     with open(devpath, "r") as fin:
#         for line in fin.readlines():
#             sentence = re.sub('\-\[[A-Z]*\]',  '', line)
#             clean_sentences.append(sentence)
#     return clean_sentences


# new data handling:
def import_jsons_to_df(dataset_path, filename):
    fp = f"{dataset_path}/{filename}.jsonl"
    tagged_data = []
    with jsonlines.open(fp, 'r') as f:
        for line in f:
            for token in line["sent_items"]:
                tagged_data.append([line["id"], token[0], token[1]])
    return pd.DataFrame(tagged_data, columns=["sentence_id", "words", "labels"])


# def untagged_sentences(fp):
#     clean = []
#     with jsonlines.open(fp, "r") as f:
#         for line in f:
#             words = [w[0] for w in line["sent_items"]]
#             clean.append(" ".join(words))
#     return clean


def configure_model(args, model_type="roberta", model_name="roberta-base"):
    model = NERModel(
        model_type,
        model_name,
        args=args
    )
    return model


def main():
    model_args = set_args()
    outputs = "./outputs"
    dev_set = import_jsons_to_df(args.dataset_path, DEV)
    train_set = import_jsons_to_df(args.dataset_path, TRAIN)
    model = configure_model(args=model_args)
    model.train_model(train_set, eval_data=dev_set)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', help='', default="64")
    parser.add_argument('--epochs', help='', default="3")
    parser.add_argument('--wandb_project', help='', default="only_hearst_uniques")
    parser.add_argument('--experiment_suffix', help='', default="manual")
    parser.add_argument('--dataset_path', help='', default="../data/musicians_dataset")
    parser.add_argument('--version_name', help='', default="all_without_person")
    parser.add_argument('--suffix', help='', default="_unique")
    parser.add_argument('--target_tag', help='', default="MUS")
    parser.add_argument('--superclass_tag', help='', default="PER")
    parser.add_argument('--experiments_path', help='', default='../experiments')

    args = parser.parse_args()
    BATCH_SIZE = int(args.batch_size)
    EPOCHS = int(args.epochs)
    
    if args.experiment_suffix == "manual":
        DEV, TRAIN = f"dev_converted", f"split_train_{args.version_name}"
    else:
        DEV, TRAIN = f"split_dev_{args.version_name}", f"split_train_{args.version_name}"
    if args.superclass_tag:
        LABELS = [f"B-{args.target_tag}",
                  f"B-{args.superclass_tag}",
                  f"I-{args.target_tag}",
                  f"I-{args.superclass_tag}",
                  "O"]
    else:
        LABELS = [f"B-{args.target_tag}",
                  f"I-{args.target_tag}",
                  "O"]
    best_model_dir = f"{args.experiments_path}/{args.wandb_project}-{args.experiment_suffix}/best_model"
    main()