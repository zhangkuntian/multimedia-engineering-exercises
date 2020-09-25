import os
import numpy as np
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl

from options import MODEL_CLASSES


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


class AlbertJaqketFinetuner(pl.LightningModule):

    def __init__(self, args):
        super(AlbertJaqketFinetuner, self).__init__()
        self.args = args
        config_class, model_class, tokenizer_class = MODEL_CLASSES['albert']

        self.config = config_class.from_pretrained(
            args.model_name_or_path,
            num_labels=args.num_options,
            finetuning_task=args.task_name
        )
        self.tokenizer = tokenizer_class.from_pretrained(
            args.model_name_or_path
        )
        self.model = model_class.from_pretrained(
            args.model_name_or_path,
            config=self.config
        )
        self.t_total = args.max_steps

    def forward(self, **x):
        return self.model(**x)

    def training_step(self, batch, batch_idx):
        batch = tuple(t.to(self.args.device) for t in batch)
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2],
            'labels': batch[3],
        }
        outputs = self(**inputs)

        loss = outputs[0]
        if self.args.n_gpu > 1:
            # mean() to average on multi-gpu parallel training
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        # (auto) backward gradient
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       self.args.max_grad_norm)

        return {'loss': loss, 'log': {'loss': loss}}

    def validation_step(self, batch, batch_idx):
        batch = tuple(t.to(self.args.device) for t in batch)
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2],
            'labels': batch[3],
        }
        outputs = self(**inputs)
        val_loss, logits = outputs[:2]

        preds = logits.detach().cpu().numpy()
        preds = np.argmax(preds, axis=1)
        out_label_ids = inputs['labels'].detach().cpu().numpy()
        val_acc = simple_accuracy(preds, out_label_ids)

        progress_bar = {'val_loss': val_loss, 'val_acc': val_acc}
        log = {'val_loss': val_loss, 'val_acc': val_acc}

        return {'val_loss': val_loss, 'val_acc': val_acc,
                'progress_bar': progress_bar, 'log': log}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_val_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        progress_bar = {'avg_val_loss': avg_loss, 'avg_val_acc': avg_val_acc}
        return {'avg_val_loss': avg_loss, 'avg_val_acc': avg_val_acc,
                'progress_bar': progress_bar}

    def test_step(self, batch, batch_idx):
        batch = tuple(t.to(self.args.device) for t in batch)
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2],
            'labels': batch[3],
        }
        outputs = self(**inputs)
        _, logits = outputs[:2]

        preds = logits.detach().cpu().numpy()
        preds = np.argmax(preds, axis=1)
        out_label_ids = inputs['labels'].detach().cpu().numpy()
        test_acc = simple_accuracy(preds, out_label_ids)

        return {'test_acc': test_acc, 'pred': preds, 'label': out_label_ids}

    def test_epoch_end(self, outputs):
        avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
        all_preds = np.concatenate((x['pred'] for x in outputs), axis=0)
        all_labels = np.concatenate((x['label'] for x in outputs), axis=0)

        output_eval_file = os.path.join(self.args.output_dir,
                                        'is_test_true_output_labels.txt')
        with open(output_eval_file, 'w') as fp:
            for pred, out_label_id in zip(all_preds, all_labels):
                fp.write('{} {}\n'.format(pred, out_label_id))

        output_eval_file = os.path.join(self.args.output_dir,
                                        'is_test_true_eval_results.txt',)
        with open(output_eval_file, 'w') as writer:
            writer.write('model=%s\n' % str(self.args.model_name_or_path))
            writer.write('acc  =%s\n' % str(avg_test_acc))

        progress_bar = {'avg_test_acc': avg_test_acc}

        return {'test_acc': avg_test_acc, 'log': progress_bar,
                'progress_bar': progress_bar}

    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay
             },
            {'params': [p for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0
             },
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.args.learning_rate,
                          eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.t_total
        )
        return {'optimizer': optimizer, 'scheduler': scheduler}

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass
