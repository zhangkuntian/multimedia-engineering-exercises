"""
Copied from the following repository:
    https://github.com/cl-tohoku/JAQKET_baseline
"""


import json
import logging
import os
from typing import List
from io import open
import gzip

import torch
from torch.utils.data import (
    DataLoader,
    RandomSampler,
    SequentialSampler,
    TensorDataset
)
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedTokenizer


logger = logging.getLogger(__name__)


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
        for line in self._read_json_gzip(
                os.path.join(data_dir, entities_fname)
        ):
            entity = json.loads(line.strip())
            entities[entity["title"]] = entity["text"]

        return entities

    def get_examples(self, mode, data_dir, fname,
                     entities_fname, num_options=20):
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
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            "10", "11", "12", "13", "14", "15", "16", "17", "18", "19",
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

            if answer not in options:
                continue

            if len(options) != num_options:
                skip_examples += 1
                continue

            contexts = [entities[options[i]] for i in range(num_options)]
            truth = str(options.index(answer))

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
            if "num_truncated_tokens" in inputs \
                    and inputs["num_truncated_tokens"] > 0:
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

        label = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("qid: {}".format(example.example_id))
            for (choice_idx, (input_ids, attention_mask, token_type_ids),) \
                    in enumerate(choices_features):

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


processors = {"jaqket": JaqketProcessor}
# MULTIPLE_CHOICE_TASKS_NUM_LABELS = {"jaqket", 20}


def select_field(features, field):
    return [
        [choice[field] for choice in feature.choices_features]
        for feature in features
    ]


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
    all_input_ids = torch.tensor(
        select_field(features, "input_ids"), dtype=torch.long
    )
    all_input_mask = torch.tensor(
        select_field(features, "input_mask"), dtype=torch.long
    )
    all_segment_ids = torch.tensor(
        select_field(features, "segment_ids"), dtype=torch.long
    )
    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)

    dataset = TensorDataset(
        all_input_ids, all_input_mask, all_segment_ids, all_label_ids
    )
    return dataset


def build_dataloader(args, tokenizer):
    """
    Prepare train, valid, and test dataloaders (editor: yoshinaka)

    Parameters:
    ----------
    args : argparse.ArgumentParser
    tokeniser : transformers.PreTrainedTokenizer

    Returns:
    ----------
    train_dataloader : torch.utils.data.DataLoader
    val_dataloader : torch.utils.data.DataLoader
    test_dataloader : torch.utils.data.DataLoader
    """

    train_dataset = load_and_cache_examples(
        args, args.task_name, tokenizer, evaluate=False)
    val_dataset = load_and_cache_examples(
        args, args.task_name, tokenizer, evaluate=True, test=False)
    test_dataset = load_and_cache_examples(
        args, args.task_name, tokenizer, evaluate=False, test=True)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)

    train_sampler = (RandomSampler(train_dataset)
                     if args.local_rank == -1
                     else DistributedSampler(train_dataset))
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size)
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset,
                                sampler=val_sampler,
                                batch_size=args.eval_batch_size)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset,
                                 sampler=test_sampler,
                                 batch_size=args.eval_batch_size)

    return train_dataloader, val_dataloader, test_dataloader
