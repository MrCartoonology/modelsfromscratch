import copy as cp
import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from torch.utils.tensorboard import SummaryWriter
from modelsfromscratch.setup import RunTracker


PROMPTS = dict(
    pytorch_fn="""
def lp_pool3d(
    input: Tensor,
    norm_type: Union[int, float],
    kernel_size: BroadcastingList3[int],
    stride: Optional[BroadcastingList3[int]] = None,
    ceil_mode: bool = False,
) -> Tensor:
""" ,
    my_attn_fn="""
class MultiHeadAttnWithRoPE(nn.Module):
    def __init__(
        self, model_dim: int, num_heads: int, rope_encoder: RotationalPositionalEncoding
    ):
        super(MultiHeadAttnWithRoPE, self).__init__()
        assert model_dim % num_heads == 0, "model dim must be divisible by n_heads"
        self.model_dim = model_dim
        self.num_heads = num_heads
        self.rope_encoder = rope_encoder

        self.head_dim = model_dim // num_heads

        self.Q = nn.Linear(model_dim, model_dim)
        self.K = nn.Linear(model_dim, model_dim)
        self.V = nn.Linear(model_dim, model_dim)
        self.output_proj = nn.Linear(model_dim, model_dim)

    def forward(self, X):
        _, S, _ = X.size()
        q = self.rope_encoder(self.Q(X))  # B x S x D
 """
    )

def prompt_model(res: RunTracker, global_step: int) -> Dict[str, str]:
    global PROMPTS
    cfg = res.cfg["prompt"]
    model = res.model


    prompt_result = dict(global_step=global_step)
    with torch.no_grad():
        for name, prompt in PROMPTS.items():
            print("======================================================")
            print(f"-- prompt={name} -- global_step - {global_step} -----")
            print("original prompt:\n")
            print(prompt)
            
            inputs = res.tokenizer.encode(prompt, return_tensors="pt").to("mps")
            softmax_fn = nn.Softmax(dim=0)
            step_prompt = cp.deepcopy(prompt)
            for step in range(cfg["num_tokens"]):
                logits = model(inputs)
                next_token_logits = logits[:, -1, :].flatten()  # view
                next_token_logits /= cfg["temperature"]
                probs = softmax_fn(next_token_logits)
                index = torch.multinomial(probs, num_samples=1)
                inputs = torch.concat([inputs, index.unsqueeze(0)], dim=1)
            print(f"------ REPLY: {cfg['num_tokens']} tokens, temperature={cfg['temperature']} ------")
            reply = res.tokenizer.decode(inputs[0])
            print(reply)
            prompt_result[name] = dict(prompt=prompt, reply=reply, global_step=global_step)
    return prompt_result


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: str,
    summary_writer: SummaryWriter,
    global_step: int,
    cfg: dict,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    num_batches = 0

    checksum = 0
    with torch.no_grad():
        steps = 0
        for batch in dataloader:
            steps += 1
            x, y = batch
            if cfg["compute_checksum"]:
                checksum += x.sum().item() + y.sum().item()
            x = x.to(device)
            y = y.to(device)

            pred = model(x)
            loss_value = loss_fn(pred.view(-1, pred.size(-1)), y.view(-1))
            total_loss += loss_value.item()
            num_batches += 1

            if steps >= cfg["steps_for_eval"]:
                break
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    summary_writer.add_scalar("loss", avg_loss, global_step)
    model.train()  # Switch back to training mode
    return avg_loss, checksum
