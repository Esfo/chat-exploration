from dataclasses import asdict, dataclass, fields
from collections import deque
import json
import math
import os
import time
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from read_paragraphs import read_paragraphs


@dataclass
class Config:
    # all required user-facing values are still supplied by main.py
    context_length: int
    d_model: int
    n_layers: int
    head_dim: int
    mlp_multiplier: float
    tokenpath: str
    textsource: str
    batch_size: int
    learning_rate: float
    log_every: int

    # saving / resuming / stopping
    model_output: str = ""
    resume_from: str = ""
    target_loss: float = 0.0
    target_window: int = 100
    checkpoint_every: int = 200
    max_steps: int = 5000

    # modern training controls
    device: str = "auto"  # auto, cuda, cpu
    dtype: str = "auto"  # auto, float32, float16, bfloat16
    compile_model: bool = True
    optimizer: str = "adamw"
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0

    # learning-rate scheduling. AdamW is already adaptive per-parameter; the
    # scheduler controls the global LR. cosine is the standard default, plateau
    # adapts to validation loss when validation is enabled.
    scheduler: str = "cosine"  # cosine, plateau, constant
    warmup_steps: int = 100
    min_learning_rate: float = 3e-5
    plateau_factor: float = 0.5
    plateau_patience: int = 3

    # data loading / validation
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    validation_fraction: float = 0.05
    eval_every: int = 200
    eval_batches: int = 20
    seed: int = 1

    # architecture switches
    norm_type: str = "rmsnorm"  # rmsnorm or layernorm
    use_rope: bool = True
    use_swiglu: bool = True
    tie_weights: bool = True
    dropout: float = 0.0

    # generation / sampling controls
    sample_max_new_tokens: int = 100
    sample_temperature: float = 0.8
    sample_top_k: int = 50
    sample_top_p: float = 0.95
    sample_greedy: bool = False

    # filled in by train() / load_model()
    vocab_size: int = 0

    @property
    def n_heads(self):
        assert self.d_model % self.head_dim == 0
        return self.d_model // self.head_dim

    @property
    def d_ff(self):
        return int(self.d_model * self.mlp_multiplier)


@dataclass
class VocabTokenizer:
    token_to_id: dict
    id_to_token: dict
    match_tokens: list
    vocab_size: int
    pad_id: int = 0
    eos_id: int = 1

    def encode(self, text):
        token_ids = []
        i = 0
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue

            match = None
            for token in self.match_tokens:
                if text.startswith(token, i):
                    match = token
                    break

            if match is None:
                raise ValueError(f"unknown token near: {text[i:i + 30]!r}")

            token_ids.append(self.token_to_id[match])
            i += len(match)

        token_ids.append(self.eos_id)
        return token_ids

    def decode(self, token_ids):
        tokens = []
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id == self.pad_id or token_id == self.eos_id:
                continue
            tokens.append(self.id_to_token[token_id])
        return " ".join(tokens)


def build_tokenizer_from_jsonl(path):
    token_to_id = {"<PAD>": 0, "<EOS>": 1}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line == "":
                continue
            token = json.loads(line)
            if token and token not in token_to_id:
                token_to_id[token] = len(token_to_id)

    id_to_token = {token_id: token for token, token_id in token_to_id.items()}
    match_tokens = sorted(
        [token for token in token_to_id if token not in ("<PAD>", "<EOS>")],
        key=len,
        reverse=True,
    )
    return VocabTokenizer(token_to_id, id_to_token, match_tokens, len(token_to_id))


def _config_from_dict(data):
    names = {field.name for field in fields(Config)}
    filtered = {key: value for key, value in data.items() if key in names}
    return Config(**filtered)


def resolve_device(device):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU", flush=True)
        return torch.device("cpu")
    return torch.device(device)


def resolve_dtype(dtype, device):
    if device.type != "cuda":
        return torch.float32
    if dtype == "auto":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]


class TokenChunkDataset(IterableDataset):
    def __init__(self, paragraphs, tokenizer, context_length, seed=1, shuffle=True):
        self.paragraphs = list(paragraphs)
        self.tokenizer = tokenizer
        self.context_length = context_length
        self.seed = seed
        self.shuffle = shuffle

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker.id
            num_workers = worker.num_workers

        indices = list(range(worker_id, len(self.paragraphs), num_workers))
        rng = torch.Generator()
        epoch = 0
        buffer = []

        while True:
            if self.shuffle and len(indices) > 1:
                rng.manual_seed(self.seed + epoch * 9973 + worker_id)
                order = torch.randperm(len(indices), generator=rng).tolist()
                iterable_indices = [indices[i] for i in order]
            else:
                iterable_indices = indices

            for index in iterable_indices:
                try:
                    buffer.extend(self.tokenizer.encode(self.paragraphs[index]))
                except ValueError:
                    continue

                while len(buffer) >= self.context_length + 1:
                    chunk = buffer[: self.context_length + 1]
                    buffer = buffer[self.context_length:]
                    x = torch.tensor(chunk[:-1], dtype=torch.long)
                    y = torch.tensor(chunk[1:], dtype=torch.long)
                    yield x, y
            epoch += 1


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return self.weight * x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


def make_norm(dim, norm_type):
    if norm_type == "rmsnorm":
        return RMSNorm(dim)
    if norm_type == "layernorm":
        return nn.LayerNorm(dim)
    raise ValueError(f"unknown norm_type={norm_type!r}")


def precompute_rope_frequencies(context_length, head_dim, base=10000.0):
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even when RoPE is enabled")
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    positions = torch.arange(context_length).float()
    freqs = torch.outer(positions, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: batch, heads, seq, head_dim
    seq_len = x.size(-2)
    cos = cos[:seq_len].to(device=x.device, dtype=x.dtype)[None, None, :, :]
    sin = sin[:seq_len].to(device=x.device, dtype=x.dtype)[None, None, :, :]
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.d_model = cfg.d_model
        self.use_rope = cfg.use_rope
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        if self.use_rope:
            cos, sin = precompute_rope_frequencies(cfg.context_length, cfg.head_dim)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        if self.use_rope:
            q = apply_rope(q, self.rope_cos, self.rope_sin)
            k = apply_rope(k, self.rope_cos, self.rope_sin)

        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = cfg.d_ff
        self.w12 = nn.Linear(cfg.d_model, 2 * hidden, bias=False)
        self.w3 = nn.Linear(hidden, cfg.d_model, bias=False)

    def forward(self, x):
        x_value, x_gate = self.w12(x).chunk(2, dim=-1)
        return self.w3(x_value * F.silu(x_gate))


class ReLUMlp(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.ReLU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn_norm = make_norm(cfg.d_model, cfg.norm_type)
        self.attn = CausalSelfAttention(cfg)
        self.mlp_norm = make_norm(cfg.d_model, cfg.norm_type)
        self.mlp = SwiGLU(cfg) if cfg.use_swiglu else ReLUMlp(cfg)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))
        x = x + self.mlp(self.mlp_norm(x))
        return x


class LanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position_embedding = None if cfg.use_rope else nn.Embedding(cfg.context_length, cfg.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.final_norm = make_norm(cfg.d_model, cfg.norm_type)
        self.output = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.output.weight = self.token_embedding.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        batch_size, seq_len = x.shape
        if seq_len > self.cfg.context_length:
            raise ValueError(f"sequence length {seq_len} exceeds context_length {self.cfg.context_length}")

        h = self.token_embedding(x)
        if self.position_embedding is not None:
            positions = torch.arange(seq_len, device=x.device)
            h = h + self.position_embedding(positions)[None, :, :]

        for block in self.blocks:
            h = block(h)
        h = self.final_norm(h)
        logits = self.output(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def split_paragraphs(textsource, validation_fraction, seed):
    paragraphs = list(read_paragraphs(textsource))
    if not paragraphs:
        raise ValueError(f"no training paragraphs found in {textsource!r}")
    if validation_fraction <= 0 or len(paragraphs) < 2:
        return paragraphs, []

    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(paragraphs), generator=generator).tolist()
    val_count = min(len(paragraphs) - 1, max(1, int(len(paragraphs) * validation_fraction)))
    val_indices = set(order[:val_count])
    train_paragraphs = [paragraph for index, paragraph in enumerate(paragraphs) if index not in val_indices]
    val_paragraphs = [paragraphs[index] for index in order[:val_count]]
    return train_paragraphs, val_paragraphs


def make_loader(paragraphs, tokenizer, cfg, shuffle=True):
    dataset = TokenChunkDataset(paragraphs, tokenizer, cfg.context_length, seed=cfg.seed, shuffle=shuffle)
    num_workers = min(cfg.num_workers, len(paragraphs))
    kwargs = {
        "batch_size": cfg.batch_size,
        "num_workers": num_workers,
        "pin_memory": cfg.pin_memory and torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = cfg.prefetch_factor
    return DataLoader(dataset, **kwargs)


def make_optimizer(model, cfg):
    if cfg.optimizer != "adamw":
        raise ValueError("only optimizer='adamw' is currently supported")

    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() >= 2:
            decay.append(param)
        else:
            no_decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": cfg.weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
    )


def get_lr(step, cfg):
    if cfg.scheduler == "constant":
        return cfg.learning_rate
    if cfg.warmup_steps > 0 and step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / cfg.warmup_steps
    if cfg.scheduler == "cosine":
        if cfg.max_steps <= cfg.warmup_steps:
            return cfg.learning_rate
        progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
        progress = min(1.0, max(0.0, progress))
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cfg.min_learning_rate + coeff * (cfg.learning_rate - cfg.min_learning_rate)
    return cfg.learning_rate


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


@torch.no_grad()
def evaluate(model, loader, cfg, device, amp_dtype):
    model.eval()
    losses = []
    iterator = iter(loader)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    for _ in range(cfg.eval_batches):
        x, y = next(iterator)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
            _, loss = model(x, y)
        losses.append(float(loss.item()))
    model.train()
    return sum(losses) / len(losses)


def save_model(path, model, cfg, optimizer=None, step=0, best_val_loss=None):
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "config": asdict(cfg),
        "step": step,
        "best_val_loss": best_val_loss,
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)
    print(f"model saved to {path}", flush=True)


def load_model(path, map_location="cpu"):
    checkpoint = torch.load(path, map_location=map_location)
    if "model_state" not in checkpoint:
        raise ValueError("unsupported checkpoint format: expected a PyTorch checkpoint written by save_model")
    cfg = _config_from_dict(checkpoint["config"])
    model = LanguageModel(cfg)
    model.load_state_dict(checkpoint["model_state"])
    return model, cfg, checkpoint


def load_model_for_training(path, cfg, device):
    checkpoint = torch.load(path, map_location=device)
    if "model_state" not in checkpoint:
        raise ValueError("unsupported checkpoint format: expected a PyTorch checkpoint written by save_model")
    saved_cfg = _config_from_dict(checkpoint["config"])
    if saved_cfg.vocab_size != cfg.vocab_size:
        raise ValueError(
            f"saved model vocab_size ({saved_cfg.vocab_size}) does not match current tokenizer "
            f"vocab_size ({cfg.vocab_size}). use the same tokenpath the model was trained with."
        )
    # runtime controls come from main.py; architecture/vocab-sensitive choices come from the checkpoint
    for name in (
        "context_length", "d_model", "n_layers", "head_dim", "mlp_multiplier", "vocab_size",
        "norm_type", "use_rope", "use_swiglu", "tie_weights", "dropout",
    ):
        setattr(cfg, name, getattr(saved_cfg, name))
    model = LanguageModel(cfg).to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model, checkpoint


def filter_logits(logits, temperature=1.0, top_k=0, top_p=1.0):
    logits = logits / max(temperature, 1e-6)
    if top_k and top_k > 0 and top_k < logits.numel():
        values, _ = torch.topk(logits, top_k)
        logits = logits.masked_fill(logits < values[-1], float("-inf"))
    if top_p and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = probs.cumsum(dim=-1)
        remove = cumulative > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        filtered = torch.full_like(logits, float("-inf"))
        filtered.scatter_(0, sorted_indices, sorted_logits)
        logits = filtered
    return logits


@torch.inference_mode()
def generate(
    prompt,
    tokenizer,
    model,
    cfg,
    max_new_tokens=None,
    temperature=None,
    top_k=None,
    top_p=None,
    greedy=None,
    device=None,
):
    model.eval()
    if device is None:
        device = next(model.parameters()).device
    max_new_tokens = cfg.sample_max_new_tokens if max_new_tokens is None else max_new_tokens
    temperature = cfg.sample_temperature if temperature is None else temperature
    top_k = cfg.sample_top_k if top_k is None else top_k
    top_p = cfg.sample_top_p if top_p is None else top_p
    greedy = cfg.sample_greedy if greedy is None else greedy

    ids = tokenizer.encode(prompt)
    for _ in range(max_new_tokens):
        recent_ids = ids[-cfg.context_length:]
        x = torch.tensor(recent_ids, dtype=torch.long, device=device)[None, :]
        logits, _ = model(x)
        next_logits = logits[0, -1]
        if greedy or temperature == 0:
            next_id = int(torch.argmax(next_logits).item())
        else:
            next_logits = filter_logits(next_logits, temperature=temperature, top_k=top_k, top_p=top_p)
            probs = F.softmax(next_logits, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        ids.append(next_id)
        if next_id == tokenizer.eos_id:
            break
    return tokenizer.decode(ids)


def train(cfg):
    torch.manual_seed(cfg.seed)
    tokenizer = build_tokenizer_from_jsonl(cfg.tokenpath)
    cfg.vocab_size = tokenizer.vocab_size

    device = resolve_device(cfg.device)
    amp_dtype = resolve_dtype(cfg.dtype, device)
    autocast_enabled = device.type == "cuda" and amp_dtype != torch.float32
    use_scaler = device.type == "cuda" and amp_dtype == torch.float16

    print("Training setup:")
    print(f"vocab_size     = {cfg.vocab_size}")
    print(f"context_length = {cfg.context_length}")
    print(f"d_model        = {cfg.d_model}")
    print(f"n_layers       = {cfg.n_layers}")
    print(f"head_dim       = {cfg.head_dim}")
    print(f"n_heads        = {cfg.n_heads}")
    print(f"d_ff           = {cfg.d_ff}")
    print(f"device         = {device}")
    print(f"amp_dtype      = {amp_dtype}")
    print(f"optimizer      = {cfg.optimizer}")
    print(f"scheduler      = {cfg.scheduler}")

    train_paragraphs, val_paragraphs = split_paragraphs(cfg.textsource, cfg.validation_fraction, cfg.seed)
    print(f"paragraphs     = {len(train_paragraphs)} train / {len(val_paragraphs)} validation")

    start_step = 0
    best_val_loss = None
    if cfg.resume_from:
        print(f"resuming from {cfg.resume_from}", flush=True)
        model, checkpoint = load_model_for_training(cfg.resume_from, cfg, device)
        start_step = int(checkpoint.get("step", 0))
        best_val_loss = checkpoint.get("best_val_loss")
    else:
        model = LanguageModel(cfg).to(device)
        checkpoint = {}

    raw_model = model
    optimizer = make_optimizer(raw_model, cfg)
    if cfg.resume_from and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    if cfg.compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(raw_model)
            print("torch.compile enabled", flush=True)
        except Exception as exc:
            print(f"torch.compile unavailable ({exc}); continuing without compile", flush=True)
            model = raw_model

    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)
    train_loader = make_loader(train_paragraphs, tokenizer, cfg, shuffle=True)
    train_iter = iter(train_loader)
    val_loader = make_loader(val_paragraphs, tokenizer, cfg, shuffle=False) if val_paragraphs else None

    recent_losses = deque(maxlen=cfg.target_window)
    last_log_time = time.time()
    last_log_step = start_step
    tokens_since_log = 0
    plateau_bad_evals = 0
    plateau_best = None
    plateau_lr = cfg.learning_rate

    try:
        for step in range(start_step + 1, cfg.max_steps + 1):
            if cfg.scheduler in ("cosine", "constant"):
                lr = get_lr(step - 1, cfg)
            else:
                lr = plateau_lr
            set_optimizer_lr(optimizer, lr)

            x, y = next(train_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=autocast_enabled):
                _, loss = model(x, y)

            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(raw_model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            loss_value = float(loss.item())
            recent_losses.append(loss_value)
            tokens_since_log += x.numel()

            if step == 1 or step % cfg.log_every == 0:
                now = time.time()
                elapsed = max(1e-9, now - last_log_time)
                steps_since = max(1, step - last_log_step)
                tokens_per_second = tokens_since_log / elapsed
                avg = sum(recent_losses) / len(recent_losses)
                print(
                    f"step={step} loss={loss_value:.4f} avg{len(recent_losses)}={avg:.4f} "
                    f"lr={lr:.2e} {elapsed / steps_since:.2f}s/step {tokens_per_second:.0f} tok/s",
                    flush=True,
                )
                last_log_time = now
                last_log_step = step
                tokens_since_log = 0

            if val_loader is not None and cfg.eval_every and step % cfg.eval_every == 0:
                val_loss = evaluate(model, val_loader, cfg, device, amp_dtype)
                print(f"step={step} val_loss={val_loss:.4f}", flush=True)
                if best_val_loss is None or val_loss < best_val_loss:
                    best_val_loss = val_loss
                if cfg.scheduler == "plateau":
                    if plateau_best is None or val_loss < plateau_best:
                        plateau_best = val_loss
                        plateau_bad_evals = 0
                    else:
                        plateau_bad_evals += 1
                        if plateau_bad_evals >= cfg.plateau_patience:
                            plateau_lr = max(cfg.min_learning_rate, plateau_lr * cfg.plateau_factor)
                            plateau_bad_evals = 0
                            print(f"plateau scheduler reduced lr to {plateau_lr:.2e}", flush=True)

            if cfg.model_output and step % cfg.checkpoint_every == 0:
                save_model(cfg.model_output, raw_model, cfg, optimizer=optimizer, step=step, best_val_loss=best_val_loss)

            if cfg.target_loss and len(recent_losses) == cfg.target_window:
                avg = sum(recent_losses) / cfg.target_window
                if avg <= cfg.target_loss:
                    print(
                        f"average loss over last {cfg.target_window} steps ({avg:.4f}) "
                        f"reached target {cfg.target_loss} at step {step}, stopping",
                        flush=True,
                    )
                    break
        else:
            step = cfg.max_steps
    except KeyboardInterrupt:
        print(f"\ninterrupted at step {step}", flush=True)

    if cfg.model_output:
        save_model(cfg.model_output, raw_model, cfg, optimizer=optimizer, step=step, best_val_loss=best_val_loss)

    print(generate("", tokenizer, raw_model, cfg, device=device), flush=True)


if __name__ == "__main__":
    raise SystemExit("run main.py to configure and start training")
