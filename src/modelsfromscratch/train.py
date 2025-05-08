import time
import torch
import torch.nn.utils as nn_utils
from modelsfromscratch.setup import (
    RunTracker,
    setup_training,
    setup_optim,
    save_model_and_meta
)
import modelsfromscratch.utils as ms_utils
from modelsfromscratch.eval import evaluate_model, prompt_model


def train(res: RunTracker) -> RunTracker:
    res = setup_training(res)

    model = res.model
    train_cfg = res.cfg["train"]
    eval_cfg = res.cfg["eval"]
    prompt_cfg = res.cfg["prompt"]

    device = ms_utils.get_device(res.cfg["device"])

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = setup_optim(model=model, cfg=res.cfg["optim"])
    res.optimizer = optimizer

    # Track training and validation losses
    res.train_losses = []
    res.val_losses = []
    res.prompts = []
    global_step = 0

    train_dl = res.dataloaders["train"]["dataloader"]
    val_dl = res.dataloaders["val"]["dataloader"]
    for epoch in range(train_cfg["num_epochs"]):
        t0 = time.time()
        for batch in train_dl:
            global_step += 1
            # Get batch data
            model.train()
            x, y = batch
            x = x.to(device)
            y = y.to(device)

            pred = model(x)  # [batch_size, seq_len, vocab_size]
            train_loss = loss_fn(pred.view(-1, pred.size(-1)), y.view(-1))

            optimizer.zero_grad()
            train_loss.backward()
            nn_utils.clip_grad_norm_(model.parameters(), train_cfg["max_grad_norm"])
            optimizer.step()

            res.train_losses.append((global_step, train_loss.item()))
            res.train_writer.add_scalar("loss", train_loss.item(), global_step)

            msg = f"Epoch {epoch:2d} Step {global_step:4d}: Train Loss: {train_loss.item():.4f}."

            # Evaluate periodically
            val_loss = None
            if eval_cfg["enable"] and (
                global_step % eval_cfg["train_steps_between_evals"] == 0
            ):
                val_loss, checksum = evaluate_model(
                    model=model,
                    dataloader=val_dl,
                    loss_fn=loss_fn,
                    device=device,
                    summary_writer=res.val_writer,
                    global_step=global_step,
                    cfg=eval_cfg,
                )
                res.val_losses.append((global_step, val_loss))
                msg += f" Val Loss: {val_loss:.4f} checksum: {checksum}."
            print(msg)

            # prompt periodically if loss good enough
            if (
                prompt_cfg["enable"]
                and val_loss
                and (val_loss <= prompt_cfg["max_loss"])
                and (global_step % prompt_cfg["train_steps_between_prompts"] == 0)
            ):
                res.prompts.append(prompt_model(res=res, global_step=global_step))

            if train_cfg["max_steps"] > 0 and global_step >= train_cfg["max_steps"]:
                break
        if train_cfg["max_steps"] > 0 and global_step >= train_cfg["max_steps"]:
            break
        epoch_minutes = (time.time() - t0) / 60.0
        print(f"Epoch time: {epoch_minutes:.2f} minutes. global_step: {global_step:5d}")
        save_model_and_meta(res=res, epoch=epoch)
    return res
