
import os
import random
from itertools import islice

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage, Loss

from datasets import get_celeba
from model import Glow


def check_manual_seed(seed):
    seed = seed or random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)

    print("Using seed: {seed}".format(seed=seed))


def compute_loss(nll, reduction="mean"):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "none":
        losses = {"nll": nll}

    losses["total_loss"] = losses["nll"]

    return losses


def compute_loss_y(nll, y_logits, y_weight, y, multi_class, reduction="mean"):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "none":
        losses = {"nll": nll}

    if multi_class:
        y_logits = torch.sigmoid(y_logits)
        loss_classes = F.binary_cross_entropy_with_logits(
            y_logits, y, reduction=reduction
        )
    else:
        loss_classes = F.cross_entropy(
            y_logits, torch.argmax(y, dim=1), reduction=reduction
        )

    losses["loss_classes"] = loss_classes
    losses["total_loss"] = losses["nll"] + y_weight * loss_classes

    return losses


def train(
    dataroot,
    augment,
    batch_size,
    eval_batch_size,
    epochs,
    saved_model,
    seed,
    hidden_channels,
    K,
    L,
    actnorm_scale,
    flow_permutation,
    flow_coupling,
    LU_decomposed,
    learn_top,
    y_condition,
    y_weight,
    max_grad_clip,
    max_grad_norm,
    lr,
    n_workers,
    cuda,
    n_init_batches,
    output_dir,
    saved_optimizer,
    warmup,
):
  
    image_shape = (64, 64, 3)
    num_classes = 40

    device = "cpu" if (not torch.cuda.is_available() or not cuda) else "cuda:0"

    check_manual_seed(seed)

    # Note: unsupported for now
    multi_class = False

    train_dataset, test_dataset = get_celeba(augment=augment, dataroot=dataroot)

    train_loader = DataLoader(
                              train_dataset, 
                              batch_size=batch_size, 
                              shuffle=True, 
                              num_workers=n_workers, 
                              drop_last=True
                             )


    test_loader = DataLoader(
                                test_dataset, 
                                batch_size=eval_batch_size, 
                                shuffle=False, 
                                num_workers=n_workers, 
                                drop_last=False
                                )

    model = Glow(
        image_shape,
        hidden_channels,
        K,
        L,
        actnorm_scale,
        flow_permutation,
        flow_coupling,
        LU_decomposed,
        num_classes,
        learn_top,
        y_condition,
    )


    model = model.to(device)

    optimizer = optim.Adamax(model.parameters(), lr=lr, weight_decay=5e-5)

    lr_lambda = lambda epoch: min(1.0, (epoch + 1) / warmup)  # noqa
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


    def step(engine, batch):
        model.train()
        optimizer.zero_grad()

        x, y = batch
        x = x.to(device)

        if y_condition:
            y = y.to(device)
            z, nll, y_logits = model(x, y)
            losses = compute_loss_y(nll, y_logits, y_weight, y, multi_class)
        else:
            z, nll, y_logits = model(x, None)
            losses = compute_loss(nll)

        losses["total_loss"].backward()

        if max_grad_clip > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), max_grad_clip)
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()

        return losses

    def eval_step(engine, batch):
        model.eval()

        x, y = batch
        x = x.to(device)

        with torch.no_grad():
            if y_condition:
                y = y.to(device)
                z, nll, y_logits = model(x, y)
                losses = compute_loss_y(
                    nll, y_logits, y_weight, y, multi_class, reduction="none"
                )
            else:
                z, nll, y_logits = model(x, None)
                losses = compute_loss(nll, reduction="none")

        return losses
        
    trainer = Engine(step)
    checkpoint_handler = ModelCheckpoint(
        output_dir, "glow", n_saved=2, require_empty=False
    )

    trainer.add_event_handler(
        Events.EPOCH_COMPLETED,
        checkpoint_handler,
        {"model": model, "optimizer": optimizer},
    )

    monitoring_metrics = ["total_loss"]
    RunningAverage(output_transform=lambda x: x["total_loss"]).attach(
        trainer, "total_loss"
    )

    evaluator = Engine(eval_step)

    Loss(
        lambda x, y: torch.mean(x),
        output_transform=lambda x: (
            x["total_loss"],
            torch.empty(x["total_loss"].shape[0]),
        ),
    ).attach(evaluator, "total_loss")

    if y_condition:
        monitoring_metrics.extend(["nll"])
        RunningAverage(output_transform=lambda x: x["nll"]).attach(trainer, "nll")

        Loss(
            lambda x, y: torch.mean(x),
            output_transform=lambda x: (x["nll"], torch.empty(x["nll"].shape[0])),
        ).attach(evaluator, "nll")

    pbar = ProgressBar()
    pbar.attach(trainer, metric_names=monitoring_metrics)

    # load pre-trained model if given
    if saved_model:
        model.load_state_dict(torch.load(saved_model))
        model.set_actnorm_init()

        if saved_optimizer:
            optimizer.load_state_dict(torch.load(saved_optimizer))

        file_name, ext = os.path.splitext(saved_model)
        resume_epoch = 0

        @trainer.on(Events.STARTED)
        def resume_training(engine):
            engine.state.epoch = resume_epoch
            engine.state.iteration = resume_epoch * len(engine.state.dataloader)

    @trainer.on(Events.STARTED)
    def init(engine):
        model.train()

        init_batches = []
        init_targets = []

        with torch.no_grad():
            for batch, target in islice(train_loader, None, n_init_batches):
                init_batches.append(batch)
                init_targets.append(target)

            init_batches = torch.cat(init_batches).to(device)

            assert init_batches.shape[0] == n_init_batches * batch_size

            if y_condition:
                init_targets = torch.cat(init_targets).to(device)
            else:
                init_targets = None

            model(init_batches, init_targets)

    @trainer.on(Events.EPOCH_COMPLETED)
    def evaluate(engine):
        evaluator.run(test_loader)

        scheduler.step()
        metrics = evaluator.state.metrics

        losses = ", ".join([f"{key}: {value:.2f}" for key, value in metrics.items()])

        print(f"Validation Results - Epoch: {engine.state.epoch} {losses}")

    timer = Timer(average=True)
    timer.attach(
        trainer,
        start=Events.EPOCH_STARTED,
        resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED,
        step=Events.ITERATION_COMPLETED,
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        pbar.log_message(
            f"Epoch {engine.state.epoch} done. Time per batch: {timer.value():.3f}[s]"
        )
        timer.reset()

    trainer.run(train_loader, epochs)

    torch.save(model.state_dict(), os.path.join(output_dir, "glow_checkpoint.pt"))


