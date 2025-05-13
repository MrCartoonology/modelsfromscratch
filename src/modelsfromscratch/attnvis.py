import yaml
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML, display

import torch
import torch.nn as nn

from modelsfromscratch.data import get_dataloaders_from_files, get_tokenizer
from modelsfromscratch.setup import find_root, load_config
from modelsfromscratch.models import load_model_from

CFG_FNAME = find_root() / "config/config.yaml"


def find_best_model(savedir: str):
    path = find_root() / f"{savedir}/meta.yaml"
    with open(path, "r") as f:  
        meta = yaml.unsafe_load(f)
    losses = [l for _, l in meta["val_losses"]]
    steps = [s for s, _ in meta["val_losses"]]
    best_loss = min(losses)
    best_step = steps[losses.index(min(losses))]
    best_epoch = int(round(best_step / meta["train"]["n_steps"]))
    best_model_path = f"{savedir}/model_epoch{best_epoch:02d}.pt"
    print(f"best_step: {best_step} loss {best_loss:.4f} closest epoch={best_epoch} pt={savedir}/model_epoch{best_epoch:02d}.pt")

    return meta, best_step, best_loss, best_model_path


def masked_entropies_bits(attn: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len, _ = attn.size()
    res = torch.zeros(batch_size, seq_len, device=attn.device)
    for s in range(seq_len):
        sp1 = s + 1
        probs = attn[:, s, 0:sp1]
        log_probs = torch.log2(torch.clamp(probs, 1e-9))
        res[:, s] = -(probs * log_probs).sum(dim=-1)
    return res


def num_batches(dataloaders):
    n_batches = min([len(dl['dataloader']) for _, dl in dataloaders.items()])
    n_batches = max(1, n_batches - 1)
    return n_batches


def unpack_shape(cfg: dict, dataloaders: dict):
    n_batches = num_batches(dataloaders)
    seq_len = cfg['dataloader']['seq_len']
    n_blocks = cfg['models']['transformer']['depth']
    n_heads = cfg['models']['transformer']['num_heads']
    n_ex = cfg['dataloader']['batch_size'] * n_batches
    return [n_blocks, n_heads, seq_len, n_ex]


def init_attn_support(cfg: dict, dataloaders: dict):
    shape = unpack_shape(cfg=cfg, dataloaders=dataloaders)
    return dict(train=np.zeros(shape=shape), val=np.zeros(shape=shape))


def calc_normalized_attn_support(cfg: dict, dataloaders: dict, model: nn.Module) -> np.array:
    attn_supports = init_attn_support(cfg=cfg, dataloaders=dataloaders)

    n_blocks, n_heads, seq_len, n_ex = attn_supports['train'].shape
    divisors = np.arange(1, seq_len + 1).reshape(1, 1, seq_len, 1)  # shape (1, 1, S, 1)
    batch_size = cfg['dataloader']['batch_size']

    model.eval()
    with torch.no_grad():
        for split, attn_support in attn_supports.items():
            dl = dataloaders[split]['dataloader']
            for batch_idx, batch in enumerate(dl):
                if (batch_idx + 1) * batch_size > n_ex:
                    continue
                print(f"calc attn support split={split:8s} batch={batch_idx:5d}")
                x, y = batch
                x = x.to(cfg['device'])
                y = y.to(cfg['device'])
                logits, attn = model(x, return_attn=True)
                for block in range(n_blocks):
                    for head in range(n_heads):
                        bh_attn = attn[block][head]
                        entropies = masked_entropies_bits(attn=bh_attn)
                        na = batch_idx * batch_size
                        nb = (batch_idx + 1) * batch_size
                        attn_support[block, head, :, na:nb] = 2 ** entropies.T.detach().to('cpu')
            attn_support /= divisors

    return attn_supports


def plot_attn_weights(attn_supports):
    plt.clf()
    for split, attn_supports in attn_supports.items():
        mu = np.mean(attn_supports.flatten())
        plt.hist(attn_supports.flatten(), bins=1000, label=f'{split} mean={mu:.3f}', alpha=0.4)
    plt.xlabel('normalized support')
    plt.ylabel('count')
    plt.legend()
    plt.title('Normalized Support Attn Weights all Blocks/Heads')
    plt.show()


def plot_attn_weights_vs_seq_len(attn_supports):
    plt.clf()
    for split, attn_supports in attn_supports.items():
        means = attn_supports.mean(axis=(0, 1, 3))  # Shape: (512,)
        stds = attn_supports.std(axis=(0, 1, 3))    # Shape: (512,)
        x = np.arange(len(means))
        plt.errorbar(x, means, yerr=stds, fmt='-o', markersize=2, linewidth=1, label=split, alpha=0.5)

    plt.legend()
    plt.title('Normalized Attn Support vs Seq Len (all blocks/heads)')
    plt.tight_layout()
    plt.show()


def plot_attn_weights_per_transformer_block_with_seq_len_filter(attn_supports, seq_len_interval):
    plt.clf()
    sa, sb = seq_len_interval
    attn_supports = dict(train=attn_supports['train'][:, :, sa:sb, :],
                         val=attn_supports['val'][:, :, sa:sb, :])
    n_blocks, n_heads, _, _ = attn_supports['train'].shape

    # Plot attention support histograms by block/head for the selected seq_len range
    fig, axs = plt.subplots(n_blocks, n_heads, figsize=(12, 8), sharex=True, sharey=True)
    for block in range(n_blocks):
        for head in range(n_heads):
            ax = axs[block, head]
            train_data = attn_supports['train'][block, head].flatten()
            val_data = attn_supports['val'][block, head].flatten()
            ax.hist(train_data, bins=100, alpha=0.5, label='train')
            ax.hist(val_data, bins=100, alpha=0.5, label='val')
            ax.set_title(f"Block {block}, Head {head}")
            if block == n_blocks - 1:
                ax.set_xlabel("Support")
            if head == 0:
                ax.set_ylabel("Count")
    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right')
    fig.suptitle(f"Attention Support Histograms by Block/Head filtered to seq_len in [{sa}, {sb})", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def visualize_attention(tokenizer, ids, attention, label_id):
    """
    Display tokens with background color intensity proportional to attention weights.
    
    Args:
        tokenizer: a Hugging Face tokenizer instance
        ids (List[int]): List of token IDs (length N+1)
        attention (List[float]): List of attention weights (length N), values between 0 and 1
    """
    assert len(ids) == len(attention)
    subwords = [tokenizer.decode([id_]) for id_ in ids]
    alpha = np.clip((np.array(attention) / np.max(attention)), 0, 1)

    spans = []
    for subword, att in zip(subwords, alpha):
        # Escape angle brackets and ampersands
        safe_subword = (subword.replace("&", "&amp;")
                                .replace("<", "&lt;")
                                .replace(">", "&gt;"))
        spans.append(
            f'<span style="background-color: rgba(255,0,0,{att:.2f}); '
            'padding:1px; margin:0px; border-radius:2px;">'
            f'{safe_subword}</span>'
        )

   # Add label token with yellow background
    label_token = tokenizer.decode([label_id])
    label_token = (label_token.replace("&", "&amp;")
                               .replace("<", "&lt;")
                               .replace(">", "&gt;"))
    spans.append(
        f'<span style="background-color: rgba(255,255,0,0.7); '
        'padding:1px; margin-left:5px; border-radius:2px;">'
        f'{label_token}</span>'
    )
    html = ''.join(spans)
    display(HTML(f"<div style='white-space:pre-wrap; font-family:monospace; font-size: 14px'>{html}</div>"))


def visualize_example(split, seq_len, tokenizer, ids, attn_heads, label_id):
    plt.figure(figsize=(12, 6))
    for ii, h in enumerate(attn_heads):
        plt.plot(h, label=f'head {ii}')
        plt.legend()
    plt.title(f"Random {split} Example of {seq_len} Tokens Input Layer Attention distributions");
    plt.xlabel("past sequence token")
    plt.show()

    for ii, head in enumerate(attn_heads):
        display(HTML(f"<h2>Head {ii}</h2>"))
        visualize_attention(tokenizer=tokenizer, ids=ids, attention=head, label_id=label_id)


def load_best_model_and_data():
    savedir = "saved_models/20250512-1233"    # this is with rope fix
    meta, _, _, model_pth = find_best_model(savedir)
    cfg = load_config(CFG_FNAME)
    tokenizer = get_tokenizer(cfg=cfg)

    dbg = 300   # val has 272 files
    train_files = meta['train']['files'][0:dbg]
    val_files = meta['val']['files'][0:dbg]
    print(f"loading data loaders with saved model train/val using {dbg} files each")
    dataloaders = get_dataloaders_from_files(cfg=cfg['dataloader'], train_files=train_files, val_files=val_files, tokenizer=tokenizer)

    model = load_model_from(cfg=cfg, tokenizer=tokenizer)
    print("loading best model weights")
    model.load_state_dict(torch.load(model_pth)['model_state_dict'])
    return cfg, tokenizer, dataloaders, model


def vis_attn_over_tokens(split='train', batch_to_use=3, ex=24, seq_len=50):
    cfg, tokenizer, dataloaders, model = load_best_model_and_data()
    print("loaded model")
    dl = dataloaders[split]['dataloader']
    print("loaded dataloder")
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dl):
            if batch_idx != batch_to_use:
                continue
            x, y = batch
            x = x.to(cfg['device'])
            y = y.to(cfg['device'])
            print(f"evaluating model on batch={batch_idx} of split={split}")
            logits, attn = model(x, return_attn=True)
            seq_len_p1 = seq_len + 1
            ids = x.to('cpu').detach().numpy()[ex][0:seq_len_p1]
            attn_blk1 = [attn[0][ii][ex][seq_len][0:seq_len].to('cpu').detach().numpy() for ii in range(len(attn[0]))]
            return visualize_example(split=split, seq_len=seq_len, tokenizer=tokenizer, ids=ids[0:seq_len], attn_heads=attn_blk1, label_id=ids[seq_len])
    print("done")        


def calc_diversities_on_batch(heads, seq_len):
    cdfs = []
    for head in heads:
        dist = head[:, seq_len, 0:seq_len].to('cpu').detach().numpy().astype(np.float64)
        nc = np.sum(dist, axis=1)
        dist /= np.reshape(nc, [-1, 1])
        cdfs.append(np.cumsum(dist, axis=1))

    emds = []
    for ii, hA in enumerate(cdfs):
        for jj in range(ii+1, len(cdfs)):
            hB = cdfs[jj]
            emds.append(np.sum(np.abs(hA - hB), axis=1))
    return np.mean(np.stack(emds), axis=0)


def attention_diversity(split='train', seq_len=50, layers=[0]):
    cfg, tokenizer, dataloaders, model = load_best_model_and_data()
    print("loaded model")
    dl = dataloaders[split]['dataloader']
    print("loaded dataloder")
    model.eval()
    attn_diversities = []

    lowest = dict(score=None, ids=None)
    highest = dict(score=None, ids=None)

    def update_low_high(batch_diversities, lowest, highest, x):
        min_idx = np.argmin(batch_diversities)
        max_idx = np.argmax(batch_diversities)
        min_div = batch_diversities[min_idx]
        max_div = batch_diversities[max_idx]

        if None is lowest['score'] or min_div < lowest['score']:
            lowest['score'] = min_div
            lowest['ids'] = x.to('cpu').detach().numpy()[min_idx, 0:seq_len+1]

        if None is highest['score'] or max_div > highest['score']:
            highest['score'] = max_div
            highest['ids'] = x.to('cpu').detach().numpy()[max_idx, 0:seq_len+1]
        return lowest, highest

    with torch.no_grad():
        for batch_idx, batch in enumerate(dl):
            x, y = batch
            x = x.to(cfg['device'])
            y = y.to(cfg['device'])
            print(f"evaluating model on batch={batch_idx} of split={split}")
            logits, attn = model(x, return_attn=True)
            layers_div = []
            for lyr, heads in enumerate(attn):
                if lyr in layers:
                    layers_div.append(calc_diversities_on_batch(heads=heads, seq_len=seq_len))
            assert len(layers_div) > 0
            batch_diversities = np.mean(np.stack(layers_div), axis=0)
            lowest, highest = update_low_high(batch_diversities, lowest, highest, x)
            attn_diversities.append(batch_diversities)
    lowest['str'] = tokenizer.decode(lowest['ids'])
    highest['str'] = tokenizer.decode(highest['ids'])
    return np.concatenate(attn_diversities), lowest, highest
    

def run(do_plots=True):
    """Used for normalized entropy analysis"""
    cfg, tokenizer, dataloaders, model = load_best_model_and_data()
#    savedir = "saved_models/20250507-1427"   # This is with rope bug
    attn_supports = calc_normalized_attn_support(cfg=cfg, dataloaders=dataloaders, model=model)

    if do_plots:
        plot_attn_weights(attn_supports=attn_supports)
        plot_attn_weights_vs_seq_len(attn_supports=attn_supports)
        plot_attn_weights_per_transformer_block_with_seq_len_filter(attn_supports=attn_supports, seq_len_interval=[100, 150])

    return attn_supports


if __name__ == "__main__":
    run()
