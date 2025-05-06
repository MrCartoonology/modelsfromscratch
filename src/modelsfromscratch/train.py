import torch
import torch.nn.utils as utils

from modelsfromscratch.setup import RunTracker


def train(res: RunTracker):
    model = res.model
    train_dataloader = res.dataloaders['train']
    val_dataloader = res.dataloaders['val']
    cfg = res.cfg["train"]

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    model.train()
    for epoch in range(cfg["epochs"]):
        for step, batch in enumerate(train_dataloader):
            if cfg["stop_after_two_steps"] and step >= 2:
                break
            x, y = batch
            pred = model(x)  # [batch_size, seq_len, vocab_size]
            # Reshape pred and y for loss calculation
            # pred = [batch_size * seq_len, vocab_size]
            # y = [batch_size * seq_len]
            loss_value = loss(pred.view(-1, pred.size(-1)), y.view(-1))
            optimizer.zero_grad()
            loss_value.backward()
            utils.clip_grad_norm_(model.parameters(), cfg["max_grad_norm"])
            optimizer.step()
            print(f"Epoch {epoch}: step {step} loss: {loss_value.item()}")
