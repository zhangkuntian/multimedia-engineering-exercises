import os
import argparse
import torch
# from pytorch_lightning import Trainer
from transformers import (
    AutoConfig,
    AutoModelForMultipleChoice,
    AutoTokenizer,
)
from data_processing import processors


ALL_MODELS = {
    'ALINEAR/albert-japanese-v2': 'albert'
}
MODEL_CLASSES = {
    'albert': (AutoConfig, AutoModelForMultipleChoice, AutoTokenizer)
}


def get_args():
    """
    Prepare options using argparser
    Copied from the following repository:
        https://github.com/cl-tohoku/JAQKET_baseline

    Returns:
    ----------
    args : argparse.ArgumentParser
    """

    parser = argparse.ArgumentParser()
    # parser = Trainer.add_argparse_args(parser)

    # Required parameters
    parser.add_argument('--data_dir', default='./data', type=str, help='')
    parser.add_argument(
        '--model_type',
        default='albert',
        type=str,
        choices=(list(MODEL_CLASSES.keys())),
        help=', '.join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        '--model_name_or_path',
        default='ALINEAR/albert-japanese-v2',
        type=str,
        help=', '.join(ALL_MODELS),
    )
    parser.add_argument(
        '--task_name',
        default='jaqket',
        type=str,
        choices=('jaqket', ),
        help=', '.join(processors.keys()),
    )
    parser.add_argument(
        '--output_dir', default='./outputs/', type=str, help='')
    parser.add_argument(
        '--train_fname', default='train_questions.json', type=str, help='')
    parser.add_argument(
        '--dev_fname', default='dev1_questions.json', type=str, help='')
    parser.add_argument(
        '--test_fname', default='dev2_questions.json', type=str, help='')
    parser.add_argument(
        '--entities_fname', default='candidate_entities.json.gz', type=str,
        help='')
    # Other parameters
    parser.add_argument('--config_name', default='', type=str, help='')
    parser.add_argument('--tokenizer_name', default='', type=str, help='')
    parser.add_argument('--cache_dir', default='', type=str, help='')
    parser.add_argument('--max_seq_length', default=512, type=int, help='')
    parser.add_argument('--do_train', action='store_true', help='')
    parser.add_argument('--do_eval', action='store_true', help='')
    parser.add_argument('--do_test', action='store_true', help='')
    parser.add_argument('--evaluate_during_training',
                        action='store_true', help='')
    parser.add_argument('--num_options', default=20, type=int, help='')
    parser.add_argument('--per_gpu_train_batch_size',
                        default=8, type=int, help='')
    parser.add_argument('--per_gpu_eval_batch_size',
                        default=8, type=int, help='')
    parser.add_argument('--gradient_accumulation_steps',
                        type=int, default=1, help='')
    parser.add_argument('--learning_rate', default=5e-5, type=float, help='')
    parser.add_argument('--weight_decay', default=0.0, type=float, help='')
    parser.add_argument('--adam_epsilon', default=1e-8, type=float, help='')
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='')
    parser.add_argument('--num_train_epochs', default=3.0, type=float, help='')
    parser.add_argument('--max_steps', default=-1, type=int, help='')
    parser.add_argument('--warmup_steps', default=0, type=int, help='')

    parser.add_argument('--logging_steps', type=int, default=50, help='')
    # parser.add_argument('--save_steps', type=int, default=50, help='')
    parser.add_argument('--eval_all_checkpoints', action='store_true', help='')
    parser.add_argument('--no_cuda', action='store_true', help='')
    parser.add_argument('--overwrite_output_dir', action='store_true', help='')
    parser.add_argument('--overwrite_cache', action='store_true', help='')
    parser.add_argument('--seed', type=int, default=42, help='')
    parser.add_argument('--fp16', action='store_true', help='')
    parser.add_argument(
        '--fp16_opt_level',
        type=str,
        default='O1',
        help="['O0', 'O1', 'O2', and 'O3']."
        'See details at https://nvidia.github.io/apex/amp.html',
    )
    parser.add_argument('--local_rank', type=int, default=-1, help='')
    # # for debugging
    # parser.add_argument('--server_ip', type=str, default='', help='')
    # parser.add_argument('--server_port', type=str, default='', help='')
    # parser.add_argument('--init_global_step', type=int, default=0, help='')

    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            'Output directory ({}) already exists and is not empty. '
            'Use --overwrite_output_dir to overcome.'.format(args.output_dir)
        )

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
        )
        args.n_gpu = torch.cuda.device_count()
    else:
        # Initializes the distributed backend which will take care of
        # sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError('Task not found: %s' % (args.task_name))

    return args
