# coding=utf-8

from __future__ import absolute_import, division, print_function

import json
import tqdm
import argparse
import logging
import os
import random
import numpy as np
import torch

from typing import List
from io import open
import gzip
from torch.utils.data import (
    DataLoader,
    SequentialSampler,
    TensorDataset,
)

from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    BertJapaneseTokenizer,
    PreTrainedTokenizer,
    AutoTokenizer,
)


logger = logging.getLogger(__name__)
###############################################################################
###############################################################################


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, question, contexts, endings, label=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence
                      (context of corresponding question).
            question: string. The untokenized text of the second sequence
                      (question).
            endings: list of str. multiple choice's options.
                     Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label


class InputFeatures(object):
    def __init__(self, example_id, choices_features, label):
        self.example_id = example_id
        self.choices_features = [
            {
                "input_ids": input_ids,
                "input_mask": input_mask,
                "segment_ids": segment_ids,
            }
            for input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_examples(self, mode, data_dir, fname, entities_fname):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class JaqketProcessor(DataProcessor):
    """Processor for the Jaqket data set."""

    def _get_entities(self, data_dir, entities_fname):
        logger.info("LOOKING AT {} entities".format(data_dir))
        entities = dict()
        for line in self._read_json_gzip(os.path.join(data_dir, entities_fname)):
            entity = json.loads(line.strip())
            entities[entity["title"]] = entity["text"]

        return entities

    def get_examples(self, mode, data_dir, fname, entities_fname, num_options=20):
        """See base class."""
        logger.info("LOOKING AT {} [{}]".format(data_dir, mode))
        entities = self._get_entities(data_dir, entities_fname)
        return self._create_examples(
            self._read_json(os.path.join(data_dir, fname)),
            mode,
            entities,
            num_options,
        )

    def get_labels(self):
        """See base class."""
        return [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "10",
            "11",
            "12",
            "13",
            "14",
            "15",
            "16",
            "17",
            "18",
            "19",
        ]

    def _read_json(self, input_file):
        with open(input_file, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines

    def _read_json_gzip(self, input_file):
        with gzip.open(input_file, "rt", encoding="utf-8") as fin:
            lines = fin.readlines()
            return lines

    def _create_examples(self, lines, t_type, entities, num_options):
        """Creates examples for the training and dev sets."""

        examples = []
        skip_examples = 0

        # for line in tqdm.tqdm(
        #    lines, desc="read jaqket data", ascii=True, ncols=80
        # ):
        logger.info("read jaqket data: {}".format(len(lines)))
        for line in lines:
            data_raw = json.loads(line.strip("\n"))

            id = data_raw["qid"]
            question = data_raw["question"].replace(
                "_", "")  # "_" は cloze question
            options = data_raw["answer_candidates"][:num_options]  # TODO
            answer = data_raw["answer_entity"]

            # if answer not in options:
            #     continue

            if len(options) != num_options:
                skip_examples += 1
                continue

            contexts = [entities[options[i]] for i in range(num_options)]
            truth = str(options.index(answer)) if answer in options else None

            if len(options) == num_options:  # TODO
                examples.append(
                    InputExample(
                        example_id=id,
                        question=question,
                        contexts=contexts,
                        endings=options,
                        label=truth,
                    )
                )

        if t_type == "train":
            assert len(examples) > 1
            assert examples[0].label is not None

        logger.info("len examples: {}".format(len(examples)))
        logger.info("skip examples: {}".format(skip_examples))

        return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    logger.info("Convert examples to features")
    features = []
    # for (ex_index, example) in tqdm.tqdm(
    #    enumerate(examples),
    #    desc="convert examples to features",
    #    ascii=True,
    #    ncols=80,
    # ):
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        for ending_idx, (context, ending) in enumerate(
            zip(example.contexts, example.endings)
        ):

            text_a = context
            text_b = example.question + tokenizer.sep_token + ending
            # text_b = tokenizer.sep_token + ending

            inputs = tokenizer.encode_plus(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
                truncation="only_first",  # 常にcontextをtruncate
            )
            if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
                logger.info(
                    "Attention! you are cropping tokens (swag task is ok). "
                    "If you are training ARC and RACE and you are poping "
                    "question + options,"
                    "you need to try to use a bigger max seq length!"
                )

            input_ids, token_type_ids = (
                inputs["input_ids"],
                inputs["token_type_ids"],
            )

            # The mask has 1 for real tokens and 0 for padding tokens. Only
            # real tokens are attended to.
            attention_mask = [
                1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = (
                    [0 if mask_padding_with_zero else 1] * padding_length
                ) + attention_mask
                token_type_ids = (
                    [pad_token_segment_id] * padding_length
                ) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + (
                    [0 if mask_padding_with_zero else 1] * padding_length
                )
                token_type_ids = token_type_ids + (
                    [pad_token_segment_id] * padding_length
                )

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length
            choices_features.append(
                (input_ids, attention_mask, token_type_ids))

        label = label_map[example.label] if example.label is not None else None

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("qid: {}".format(example.example_id))
            for (choice_idx, (input_ids, attention_mask, token_type_ids),) in enumerate(
                choices_features
            ):
                logger.info("choice: {}".format(choice_idx))
                logger.info("input_ids: {}".format(
                    " ".join(map(str, input_ids))))
                logger.info(
                    "attention_mask: {}".format(
                        " ".join(map(str, attention_mask)))
                )
                logger.info(
                    "token_type_ids: {}".format(
                        " ".join(map(str, token_type_ids)))
                )
                logger.info("label: {}".format(label))

        features.append(
            InputFeatures(
                example_id=example.example_id,
                choices_features=choices_features,
                label=label,
            )
        )

    return features


def load_and_cache_examples(args, task, tokenizer, evaluate=False, test=False):
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training process
        # the dataset, and the others will use the cache
        torch.distributed.barrier()

    processor = processors[task]()
    # Load data features from cache or dataset file
    if evaluate:
        cached_mode = ".".join(args.dev_fname.split(".")[0:-1])
    elif test:
        cached_mode = ".".join(args.test_fname.split(".")[0:-1])
    else:
        cached_mode = ".".join(args.train_fname.split(".")[0:-1])
    assert (evaluate is True and test is True) is False
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            cached_mode,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )

    logger.info("Loading features from cached file %s", cached_features_file)
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("find %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if evaluate:
            examples = processor.get_examples(
                "dev",
                args.data_dir,
                args.dev_fname,
                args.entities_fname,
                num_options=args.eval_num_options,
            )
        elif test:
            examples = processor.get_examples(
                "test",
                args.data_dir,
                args.test_fname,
                args.entities_fname,
                num_options=args.eval_num_options,
            )
        else:
            examples = processor.get_examples(
                "train",
                args.data_dir,
                args.train_fname,
                args.entities_fname,
                num_options=args.train_num_options,
            )
        logger.info("Training number: %s", str(len(examples)))
        features = convert_examples_to_features(
            examples,
            label_list,
            args.max_seq_length,
            tokenizer,
            pad_on_left=False,
            pad_token_segment_id=0,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s",
                        cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0:
        # Make sure only the first process in distributed training process
        # the dataset, and the others will use the cache
        torch.distributed.barrier()

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor(select_field(
        features, "input_ids"), dtype=torch.long)
    all_input_mask = torch.tensor(
        select_field(features, "input_mask"), dtype=torch.long
    )
    all_segment_ids = torch.tensor(
        select_field(features, "segment_ids"), dtype=torch.long
    )
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long) if not test else None

    if all_label_ids is not None:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    else:
        dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)
    return dataset


###############################################################################
###############################################################################


processors = {"jaqket": JaqketProcessor}

MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"jaqket", 20}

ALL_MODELS = (
    "cl-tohoku/bert-base-japanese",
    "cl-tohoku/bert-base-japanese-whole-word-masking",
    "cl-tohoku/bert-base-japanese-char",
    "cl-tohoku/bert-base-japanese-char-whole-word-masking",
    "bandainamco-mirai/distilbert-base-japanese",
    "ALINEAR/albert-japanese-v2",
)

MODEL_CLASSES = {
    "bert": (AutoConfig, AutoModelForMultipleChoice, BertJapaneseTokenizer),
    "albert": (AutoConfig, AutoModelForMultipleChoice, AutoTokenizer),
    "distilbert": (AutoConfig, AutoModelForMultipleChoice, AutoTokenizer),
}


def select_field(features, field):
    return [
        [choice[field] for choice in feature.choices_features] for feature in features
    ]


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def evaluate(args, model, tokenizer, prefix="", evaluate=True, test=False):
    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(
            args, eval_task, tokenizer, evaluate=evaluate, test=test
        )

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * \
            max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(
            eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size
        )

        # multi-gpu evaluate
        if args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        # out_label_ids = None
        for batch in tqdm.tqdm(
            eval_dataloader, ascii=True, ncols=80, desc="Evaluating"
        ):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "token_type_ids": batch[2],
                    # "labels": batch[3],
                }
                outputs = model(**inputs)
                # if not test:
                #     tmp_eval_loss, logits = outputs[:2]
                #     eval_loss += tmp_eval_loss.mean().item()
                # else:
                logits = outputs[0]

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)

        #######################################################################
        subset_ = 'train'
        if evaluate and not test:
            subset_ = 'eval'
        elif test:
            subset_ = 'test'
        output_eval_file = os.path.join(
            eval_output_dir, subset_ + "_output_logits.csv",
        )
        np.savetxt(output_eval_file, preds, delimiter=',')
        #######################################################################

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)

    return results


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default="./data", type=str, help="")
    parser.add_argument(
        "--model_type",
        default="bert",
        type=str,
        choices=(list(MODEL_CLASSES.keys())),
        help=", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default="cl-tohoku/bert-base-japanese-whole-word-masking",
        type=str,
        help=", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--task_name",
        default="jaqket",
        type=str,
        choices=("jaqket"),
        help=", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir", default="./outputs/", type=str, help="")
    parser.add_argument(
        "--train_fname", default="train_questions.json", type=str, help="")
    parser.add_argument(
        "--dev_fname", default="dev1_questions.json", type=str, help="")
    parser.add_argument(
        "--test_fname", default="dev2_questions.json", type=str, help="")
    parser.add_argument(
        "--entities_fname", default="candidate_entities.json.gz", type=str,
        help="")
    # Other parameters
    parser.add_argument("--config_name", default="", type=str, help="")
    parser.add_argument("--tokenizer_name", default="", type=str, help="")
    parser.add_argument("--cache_dir", default="", type=str, help="")
    parser.add_argument("--max_seq_length", default=512, type=int, help="")
    # parser.add_argument("--do_train", action="store_true", help="")
    parser.add_argument("--do_eval", action="store_true", help="")
    parser.add_argument("--do_test", action="store_true", help="")
    parser.add_argument("--evaluate_during_training",
                        action="store_true", help="")
    # parser.add_argument(
    #     "--do_lower_case", action='store_true', help="")
    parser.add_argument("--train_num_options", default=20, type=int, help="")
    parser.add_argument("--eval_num_options", default=20, type=int, help="")
    # parser.add_argument("--per_gpu_train_batch_size",
    #                     default=8, type=int, help="")
    parser.add_argument("--per_gpu_eval_batch_size",
                        default=8, type=int, help="")
    parser.add_argument("--gradient_accumulation_steps",
                        type=int, default=1, help="")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="")
    parser.add_argument("--num_train_epochs", default=3.0, type=float, help="")
    parser.add_argument("--max_steps", default=-1, type=int, help="")
    parser.add_argument("--warmup_steps", default=0, type=int, help="")

    parser.add_argument("--logging_steps", type=int, default=50, help="")
    parser.add_argument("--save_steps", type=int, default=50, help="")
    parser.add_argument("--eval_all_checkpoints", action="store_true", help="")
    parser.add_argument("--no_cuda", action="store_true", help="")
    parser.add_argument("--overwrite_output_dir", action="store_true", help="")
    parser.add_argument("--overwrite_cache", action="store_true", help="")
    parser.add_argument("--seed", type=int, default=42, help="")
    # parser.add_argument("--fp16", action="store_true", help="")
    # parser.add_argument(
    #     "--fp16_opt_level",
    #     type=str,
    #     default="O1",
    #     help="['O0', 'O1', 'O2', and 'O3']."
    #     "See details at https://nvidia.github.io/apex/amp.html",
    # )
    parser.add_argument("--local_rank", type=int, default=-1, help="")
    # for debugging
    parser.add_argument("--server_ip", type=str, default="", help="")
    parser.add_argument("--server_port", type=str, default="", help="")

    parser.add_argument("--init_global_step", type=int, default=0, help="")

    args = parser.parse_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which will take care of
        # sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    # Set seed
    set_seed(args)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will
        # download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    _, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        # do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    logger.info("Training/evaluation parameters %s", args)

    results = {}
    if args.do_test and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split(
                "-")[-1] if len(checkpoints) > 1 else ""
            prefix = (
                checkpoint.split(
                    "/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            )

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)

            result = evaluate(args, model, tokenizer, prefix=prefix, evaluate=False, test=False)  # for train data
            result = evaluate(args, model, tokenizer, prefix=prefix, evaluate=True, test=False)  # for dev1 data
            result = evaluate(args, model, tokenizer, prefix=prefix, evaluate=False, test=True)  # for dev2 data
            result = evaluate(args, model, tokenizer, prefix=prefix, evaluate=False, test=True)  # for LB (test) data

            result = dict((k + "_{}".format(global_step), v)
                          for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    results = main()
    for key, result in results.items():
        print(key, result)
