import signal
import IPython
import torch
import torch.nn.utils as utils

from modelsfromscratch.setup import RunTracker

DROP_INTO_IPYTHON = False


def control_c_handler(sig, frame):
    global DROP_INTO_IPYTHON
    print("\nCaught Ctrl+C! Dropping into IPython...")
    DROP_INTO_IPYTHON = True


def train(res: RunTracker):
    if res.cfg["drop_into_ipython_on_ctrl_c"]:
        print("Setting up Ctrl+C handler")
        signal.signal(signal.SIGINT, control_c_handler)

    model = res.model
    train_dataloader = res.dataloaders['train']
    cfg = res.cfg["train"]

    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    model.train()
    for epoch in range(cfg["epochs"]):
        for step, batch in enumerate(train_dataloader):
            if cfg["steps"] > 0 and step >= cfg['steps']:
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
            if DROP_INTO_IPYTHON:
                break
    if DROP_INTO_IPYTHON:
        IPython.embed()
