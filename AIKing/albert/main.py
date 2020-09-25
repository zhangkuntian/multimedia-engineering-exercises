import logging
import random
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

from data_processing import build_dataloader, InputFeatures
from options import ALL_MODELS, MODEL_CLASSES, get_args
from model import AlbertJaqketFinetuner


formatter = '%(levelname)s : %(asctime)s : %(message)s'
logging.basicConfig(level=logging.INFO, format=formatter)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    args = get_args()
    pl.seed_everything(args.seed)
    set_seed(args)

    model_name = ALL_MODELS[args.model_name_or_path]
    _, _, tokenizer_class = MODEL_CLASSES[model_name]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    train_dl, valid_dl, test_dl = build_dataloader(args, tokenizer)

    albert_finetuner = AlbertJaqketFinetuner(args)

    early_stopping = EarlyStopping('avg_val_loss', patience=3, mode='min')
    trainer = pl.Trainer(gpus=args.n_gpu,
                         early_stop_callback=early_stopping,
                         max_epochs=args.num_train_epochs)

    if args.do_train:
        trainer.fit(albert_finetuner, train_dataloader=train_dl,
                    val_dataloaders=valid_dl)

    if args.do_test:
        trainer.test(test_dataloaders=test_dl)


if __name__ == '__main__':
    main()
