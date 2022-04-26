import torch
from torch import nn
from transformers import Trainer, EvalPrediction
from transformers.modeling_utils import ModelOutput


class QuickDrawTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False):
        logits = model(inputs["pixel_values"])
        labels = inputs.get("labels")

        loss = None
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return (loss, ModelOutput(logits=logits, loss=loss)) if return_outputs else loss

# Taken from timm - https://github.com/rwightman/pytorch-image-models/blob/master/timm/utils/metrics.py
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def quickdraw_compute_metrics(p: EvalPrediction):
    if len(p.label_ids) == 0:
        # NOTE: this was not needed in https://github.com/nateraw/quickdraw-pytorch/blob/main/quickdraw.ipynb
        # but some reason the EvalPrediction.label_ids property will be empty on the last batch,
        # even with dataloader_drop_last set to True.
        return {}
    acc1, acc5 = accuracy(
        torch.tensor(p.predictions),
        torch.tensor(p.label_ids), topk=(1, 5)
    )
    return {'acc1': acc1, 'acc5': acc5}


def init_model(num_classes: int):
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(64, 128, 3, padding='same'),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(1152, 512),
        nn.ReLU(),
        nn.Linear(512, num_classes),  # num_classes was limited to 100 here
    )
    # return nn.Sequential(
    #     nn.Conv2d(1, 64, 3, padding='same'),
    #     nn.ReLU(),
    #     nn.MaxPool2d(2),
    #     nn.Conv2d(64, 128, 3, padding='same'),
    #     nn.ReLU(),
    #     nn.MaxPool2d(2),
    #     nn.Conv2d(128, 256, 3, padding='same'),
    #     nn.ReLU(),
    #     nn.MaxPool2d(2),
    #     nn.Flatten(),
    #     nn.Linear(2304, 512),
    #     nn.ReLU(),
    #     nn.Linear(512, num_classes),
    # )


if __name__ == "__main__":
    import torch
    from torch import nn
    from transformers import TrainingArguments
    from datetime import datetime
    
    from .dataset import QuickDrawDataset

    data_dir = './data'
    max_examples_per_class = 1000
    train_val_split_pct = .5

    ds = QuickDrawDataset(data_dir, max_examples_per_class, class_limit=5)
    num_classes = len(ds.classes)
    train_ds, val_ds = ds.split(train_val_split_pct)

    timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    training_args = TrainingArguments(
        output_dir=f'./outputs_20k_{timestamp}',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        report_to=['tensorboard'],  # Update to just tensorboard if not using wandb
        logging_strategy='steps',
        logging_steps=100,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        learning_rate=0.003,
        fp16=torch.cuda.is_available(),
        dataloader_drop_last=True,
        num_train_epochs=20,
        run_name=f"quickdraw-med-{timestamp}",  # Can remove if not using wandb
        warmup_steps=10000,
        save_total_limit=5,
    )

    model = init_model(num_classes=num_classes)

    trainer = QuickDrawTrainer(
        model,
        training_args,
        data_collator=ds.collate_fn,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=None,
        compute_metrics=quickdraw_compute_metrics,
    )

    # Training
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()

    # Evaluation
    eval_results = trainer.evaluate()
    trainer.log_metrics("eval", eval_results)
    trainer.save_metrics("eval", eval_results)
