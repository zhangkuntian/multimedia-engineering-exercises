from collections import defaultdict
import json
import logging
import numpy as np
import os
from pathlib import Path
import subprocess

from scipy.special import softmax
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import optuna

###############################################################################
formatter = '%(levelname)s : %(message)s'
logging.basicConfig(level=logging.INFO, format=formatter)

# prepare path to save results & logs
save_path = Path('runs/')
nof_dirs = len(list(save_path.glob('*')))
save_path = save_path / f'{nof_dirs:02}'
save_path.mkdir(parents=True, exist_ok=True)
SAVE_PATH = str(save_path)

# seed
torch.manual_seed(0)
np.random.seed(0)

MAX_EPOCHS = 50
BATCH_SIZE = 256
N_TRIALS = 500
###############################################################################


def load_outputs(data_dir: Path, _softmax=True):

    def _load_numpy_data(path):
        d = np.loadtxt(path, delimiter=',')
        if _softmax:
            d = softmax(d, axis=1)  # Softmax(logits)
        return d

    subdirs = list(data_dir.glob('*'))
    subdirs.sort()

    tmp = {'train': [], 'dev1': [], 'dev2': [], 'test': []}
    for dir in subdirs:
        if 'seq2seq' in dir.name:
            continue
        for name, outputs in tmp.items():
            path = dir / f'{name}_output_logits.csv'
            logging.info(f'loads from `{path}`')
            np_data = _load_numpy_data(str(path))
            assert np_data.shape[1] == 20, 'Size of the logits do not match.'
            outputs.append(np_data)

    data = {}
    for name, outputs in tmp.items():
        d = np.concatenate(outputs, axis=1)  # concatenate outputs
        data[name] = d

    return data


def load_labels(data_dir: Path):
    files = list(data_dir.glob('*.json'))

    all_lables = defaultdict(list)
    for f in files:
        name = f.stem.split('_')[0]
        if name == 'aio':
            name = 'test'
        logging.info(f'loads from `{f}`')
        with f.open('r') as fin:
            for line in fin:
                data_raw = json.loads(line.strip())
                options = data_raw["answer_candidates"][:20]
                answer = data_raw["answer_entity"]
                truth = options.index(answer) if answer in options else 0
                all_lables[name].append(truth)

    return all_lables


def shuffle_train_data(outputs: np.ndarray, labels: list):
    new_outputs = []
    new_labels = []
    idx = [i for i in range(20)]
    rng = np.random.default_rng(0)  # Seed(0)
    ensemble_size = outputs.shape[1] // 20

    for i, label in enumerate(labels):
        idx_prmt = list(rng.permutation(idx))
        new_labels.append(idx_prmt.index(label))
        outs = []
        for j in range(ensemble_size):
            outs.append(np.array([outputs[i][k + j * 20] for k in idx_prmt]))
        outs = np.concatenate(outs)
        new_outputs.append(outs)

    new_outputs = np.array(new_outputs)
    assert new_outputs.shape == outputs.shape, 'Size of the logits do not match.'

    return new_outputs, new_labels


class ModelOutputsData(Dataset):
    def __init__(self, outputs, lables):
        self.data = outputs
        self.labels = lables

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        out_data = torch.tensor(self.data[idx], dtype=torch.float)
        out_label = torch.tensor(self.labels[idx], dtype=torch.long)
        return out_data, out_label

###############################################################################


class EnsembleMLP(pl.LightningModule):
    def __init__(self, trial, data, labels):
        super().__init__()
        self.trial = trial  # for optuna
        n_neurons = self.trial.suggest_categorical('n_neurons', [64, 128, 256])
        dropout = trial.suggest_uniform('dropout', 0.1, 0.5)

        self.data = data
        self.labels = labels
        self.input_size = self.data['train'].shape[1]

        self.model = nn.Sequential(
            nn.Linear(self.input_size, n_neurons),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(n_neurons, 20)
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        _, preds = torch.max(logits, dim=1)
        preds = preds.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        acc = accuracy_score(preds, y)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        _, preds = torch.max(logits, dim=1)
        preds = preds.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        acc = accuracy_score(preds, y)

        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', acc, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        _, preds = torch.max(logits, dim=1)
        preds = preds.detach().cpu().numpy()
        y = y.detach().cpu().numpy()
        acc = accuracy_score(preds, y)

        self.log('test_acc', acc, prog_bar=True, logger=True)

    def configure_optimizers(self):
        return get_optimizer(self.trial, self.model)

    def train_dataloader(self):
        dataset = ModelOutputsData(self.data['train'], self.labels['train'])
        dataloader = DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=4)
        return dataloader

    def val_dataloader(self):
        dataset = ModelOutputsData(self.data['dev1'], self.labels['dev1'])
        dataloader = DataLoader(dataset, shuffle=False, batch_size=BATCH_SIZE, num_workers=4)
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()


def get_optimizer(trial, model):
    optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'SGD'])
    lr = trial.suggest_loguniform('lr', 1e-3, 1e-1)
    weight_decay = trial.suggest_uniform('weight_decay', 0.0, 0.001)
    optimizer = getattr(torch.optim, optimizer_name)(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    return optimizer


def objective(trial):
    ckpt_path = os.path.join(SAVE_PATH, f'ckpt/trial_{trial.number}')

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', mode='min', dirpath=ckpt_path)
    early_stop_callback = EarlyStopping(monitor='val_loss', mode='min', patience=3)
    logger = TensorBoardLogger(save_dir=SAVE_PATH, name='logs')

    trainer = pl.Trainer(max_epochs=MAX_EPOCHS, gpus=1, logger=logger,
                         callbacks=[checkpoint_callback, early_stop_callback],
                         progress_bar_refresh_rate=10,
                         weights_summary=None)

    model = EnsembleMLP(trial, data=data, labels=labels)
    trainer.fit(model)
    results = trainer.test(verbose=False)

    return float(results[-1]['test_acc'])


def evaluate(model, dataloader, dname: str):
    with torch.no_grad():
        test_outs = []
        for x, y in dataloader:
            logits = model(x)
            _, preds = torch.max(logits, dim=1)
            preds = preds.detach().cpu().numpy()
            y = y.detach().cpu().numpy()

            test_outs.append({'preds': preds, 'labels': y})

    preds = np.concatenate([x['preds'] for x in test_outs], axis=0)
    labels = np.concatenate([x['labels'] for x in test_outs], axis=0)

    result_path = Path(SAVE_PATH) / f'{dname}_output_labels.txt'
    with result_path.open('w') as fout:
        for pred, label in zip(preds, labels):
            fout.write("{} {}\n".format(pred, label))
    logging.info(f'save result to `{result_path}`')

    acc = accuracy_score(preds, labels)
    logging.info(f'result on {dname}: {acc}')

    return acc


data = load_outputs(Path('../outputs/'))
labels = load_labels(Path('../data/'))


def main():
    # Shuffle train data because all labels in train data are 0.
    global data, labels
    data['train'], labels['train'] = shuffle_train_data(data['train'], labels['train'])

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS)

    # show results
    logging.info('number of finished trials: {}'.format(len(study.trials)))
    trial = study.best_trial
    logging.info(f'best trial: accuracy={trial.value}')
    for key, value in trial.params.items():
        logging.info('\t{}: {}'.format(key, value))

    # best checkpoint
    ckpt_path = f'{SAVE_PATH}/ckpt/trial_{trial.number}/'
    ckpt_path = str(list(Path(ckpt_path).glob('*.ckpt'))[0])

    # prepare dataloaders
    dev1_dataset = ModelOutputsData(data['dev1'], labels['dev1'])
    dev2_dataset = ModelOutputsData(data['dev2'], labels['dev2'])
    test_dataset = ModelOutputsData(data['test'], labels['test'])
    dev1_dataloader = DataLoader(dev1_dataset, shuffle=False, batch_size=BATCH_SIZE)
    dev2_dataloader = DataLoader(dev2_dataset, shuffle=False, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=BATCH_SIZE)

    # load trained model
    model = EnsembleMLP.load_from_checkpoint(
        ckpt_path, trial=trial, data=data, labels=labels
    )
    model.eval()

    # save results & logs
    results = {'trial': trial.number}
    results['dev1'] = evaluate(model, dev1_dataloader, 'dev1')
    results['dev2'] = evaluate(model, dev2_dataloader, 'dev2')
    results['test'] = evaluate(model, test_dataloader, 'test')
    results.update(trial.params)

    results_path = Path(SAVE_PATH) / 'results.json'
    with results_path.open('w') as fout:
        json.dump(results, fout)

    cp_path = f'{SAVE_PATH}/{os.path.basename(__file__)}'
    subprocess.run(['cp', __file__, cp_path])


if __name__ == '__main__':
    main()
