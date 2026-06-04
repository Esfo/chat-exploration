from dataclasses import dataclass, field
import math
import json
import time
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from read_paragraphs import read_paragraphs


@dataclass
class Config:
    #all of these values are configured in main.py, the configuration gateway
    #they are required arguments here so training.py holds no hardcoded settings

    #=== model architecture ===

    #number of tokens per training chunk
    context_length: int

    #width of the model's internal token vector
    #means every token becomes a vector with this many numbers
    d_model: int

    #number of transformer blocks
    #each block contains:
        #attention
        #feed-forward / MLP
    n_layers: int

    #width of one attention head.
    #n_heads is calculated from: n_heads = d_model / head_dim
    head_dim: int

    #controls how wide the MLP part gets inside each transformer block
    #(multi-layer perceptron)
    mlp_multiplier: float

    #=== data ===

    #path to the token word list the tokenizer is built from
    tokenpath: str

    #corpus the training paragraphs are read from
    textsource: str

    #=== optimisation ===

    #number of chunks trained together in one update
    batch_size: int

    #how large each training update is (the peak LR the schedule warms up to)
    learning_rate: float

    #how often (in steps) to print the training loss to stdout
    log_every: int

    #=== run controls (defaults; main.py overrides) ===

    #where to write the trained model.
    #the model is saved here on the periodic checkpoint, when the loss target is
    #reached, and if training is interrupted (ctrl-c).
    #set to a path like 'model.pt' to save; leave as "" / False to skip saving.
    model_output: str = ""

    #path to an existing saved model to continue training from.
    #leave as "" / False to start from fresh random weights.
    resume_from: str = ""

    #early-stopping target. training runs until the average loss over the last
    #target_window steps drops to or below this value (there is no fixed step
    #count). set target_loss to 0 / False to train forever until interrupted.
    target_loss: float = 0.1

    #how many recent steps the average is taken over when checking target_loss.
    target_window: int = 100

    #save a checkpoint to model_output every this many steps, so progress
    #survives a crash or interruption even between the start and the target.
    checkpoint_every: int = 200

    #=== hardware / speed (the modern, GPU-oriented knobs) ===

    #where the model trains: "auto" picks cuda if a GPU is visible else cpu.
    #force "cuda" or "cpu" to override.
    device: str = "auto"

    #number of background worker processes the DataLoader uses to prepare
    #batches in parallel while the GPU is busy. 0 = load in the main process.
    #a good starting point is a handful of your CPU cores.
    num_workers: int = 4

    #mixed-precision training: do the heavy matmuls in fp16/bf16 on the GPU.
    #roughly ~2x faster and less VRAM on a modern NVIDIA card. ignored on cpu.
    use_amp: bool = True

    #=== optimiser / schedule (standard best-practice basics) ===

    #"adamw" (recommended modern default) or "sgd" (plain gradient descent,
    #closest to the original setup but far more sensitive to learning_rate).
    optimizer: str = "adamw"

    #adamw weight decay (L2-style regularisation). ignored for sgd.
    weight_decay: float = 0.1

    #clip the global gradient norm to this value before each step so a single
    #bad batch can't blow the weights up. set to 0 / False to disable.
    grad_clip: float = 1.0

    #linear LR warmup: ramp from 0 to learning_rate over this many steps, then
    #cosine-decay. set to 0 to start at full LR immediately.
    warmup_steps: int = 100

    #cosine decay horizon: LR eases from learning_rate down to min_lr over this
    #many steps, then holds at min_lr. set to 0 / False to keep a constant LR.
    lr_decay_steps: int = 5000

    #floor the cosine schedule decays to.
    min_lr: float = 1e-4

    #=== validation ===

    #fraction of the token stream held out (never trained on) to measure
    #validation loss. set to 0 to disable validation entirely.
    val_fraction: float = 0.05

    #run a validation pass every this many steps.
    val_every: int = 200

    #how many batches to average the validation loss over.
    val_batches: int = 20

    #number of possible token IDs
        #this gets set after the tokenizer builds the vocabulary from tokenpath
        #+ 1 padding token
        #+ 1 end-of-text token
        #+ however many token strings are found in tokenpath
    #the only field not set by main: it is filled in by train() after the tokenizer is built
    vocab_size: int = 0

    @property
    def n_heads(self):
        """
        calculation of attention heads
        """

        assert self.d_model % self.head_dim == 0
        return self.d_model // self.head_dim

    @property
    def d_ff(self):
        """
        width of the feed-forward / MLP hidden layer.
        """

        return int(self.d_model * self.mlp_multiplier)


@dataclass
class VocabTokenizer:
    """
    converts token text into token IDs.
    """

    #maps each token string to one token ID
    token_to_id: dict

    #maps each token ID back to one token string
    id_to_token: dict

    #tokens sorted longest-first so multi-character tokens are matched before shorter tokens
    match_tokens: list

    #number of possible token IDs
    vocab_size: int

    #not actively used here because chunks are always exactly context_length tokens
    #reserved so token ID 0 never means a real token
    #would be used to fill shorter examples if you later train variable-length chunks
    pad_id: int = 0

    #stop codon
    eos_id: int = 1

    def encode(self, text):
        """
        convert raw text into token IDs.
        """

        #list of integer IDs produced by the tokenizer
        token_ids = []

        #position inside the text string
        i = 0

        while i < len(text):
            #skip whitespace between token strings
            if text[i].isspace():
                i += 1
                continue

            match = None

            #match the longest token string found at this position
            #this lets 'ing' become one token instead of 'i' + 'n' + 'g'
            for token in self.match_tokens:
                if text.startswith(token, i):
                    match = token
                    break

            if match is None:
                raise ValueError(f"unknown token near: {text[i:i + 30]!r}")

            #append this token's fixed vocabulary ID
            token_ids.append(self.token_to_id[match])

            #move forward by the length of the matched token string
            i += len(match)

        token_ids.append(self.eos_id)

        return token_ids

    def decode(self, token_ids):
        """
        convert token IDs back into text.
        """

        tokens = []

        for token_id in token_ids:
            #ignore special tokens like pad_id=0 and eos_id=1
            if token_id == self.pad_id or token_id == self.eos_id:
                continue

            tokens.append(self.id_to_token[int(token_id)])

        return " ".join(tokens)


def build_tokenizer_from_jsonl(path):
    """
    build tokenizer vocabulary from a JSONL file.
    """

    #start with special token IDs
    token_to_id = {
        "<PAD>": 0,
        "<EOS>": 1,
    }

    #word_survival writes one surviving token per line, each line a JSON-encoded
    #string (serde_json::to_string), e.g. "ing" — not an object with a 'text' field
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "":
                continue

            #each line decodes to exactly one token string
            #don't split on whitespace: a token may itself be punctuation/whitespace
            token = json.loads(line)

            if token == "":
                continue

            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)

    #reverse lookup for decode()
    id_to_token = {
        token_id: token
        for token, token_id in token_to_id.items()
    }

    #longest-first matching prevents shorter tokens from stealing the start of longer tokens
    match_tokens = sorted(
        [
            token
            for token in token_to_id
            if token not in ("<PAD>", "<EOS>")
        ],
        key=len,
        reverse=True,
    )

    return VocabTokenizer(
        token_to_id=token_to_id,
        id_to_token=id_to_token,
        match_tokens=match_tokens,
        vocab_size=len(token_to_id),
    )


def encode_corpus(textsource, tokenizer):
    """
    read the whole corpus once and flatten it into a single 1-D tensor of token
    IDs (paragraphs separated by the EOS that encode() appends).

    doing this once up front - instead of re-tokenising while training - lets the
    DataLoader hand out chunks cheaply from many worker processes at once.
    """

    all_ids = []
    for paragraph in read_paragraphs(textsource):
        all_ids.extend(tokenizer.encode(paragraph))

    if len(all_ids) < 2:
        raise ValueError(
            f"corpus at {textsource!r} produced too few tokens ({len(all_ids)}) "
            "to train on. check the textsource path and the tokenizer."
        )

    #int32 is plenty for vocab sizes here and halves the memory of int64
    return torch.tensor(all_ids, dtype=torch.int32)


class ChunkDataset(Dataset):
    """
    serves fixed-size next-token training chunks out of a flat token tensor.

    item i is the window starting at token i:
        x = tokens[i           : i + context_length]
        y = tokens[i + 1       : i + context_length + 1]   (x shifted by one)

    overlapping windows + DataLoader shuffling gives dense, well-mixed coverage
    of the corpus, and __getitem__ is a cheap slice so num_workers scales well.
    """

    def __init__(self, tokens, context_length):
        self.tokens = tokens
        self.context_length = context_length

    def __len__(self):
        #every start position that still leaves a full window + 1 target token
        return max(0, len(self.tokens) - self.context_length - 1)

    def __getitem__(self, i):
        chunk = self.tokens[i: i + self.context_length + 1].long()
        return chunk[:-1], chunk[1:]


def make_loaders(tokens, cfg):
    """
    split the token stream into train/validation and wrap each in a DataLoader.

    the tail val_fraction of the corpus is held out and never trained on, so the
    validation loss reflects text the model hasn't memorised.
    """

    n = len(tokens)
    if cfg.val_fraction and cfg.val_fraction > 0:
        n_val = int(n * cfg.val_fraction)
        #keep enough val tokens to form at least one window
        n_val = max(n_val, cfg.context_length + 1) if n_val else 0
    else:
        n_val = 0

    split = n - n_val
    train_tokens = tokens[:split]
    val_tokens = tokens[split:] if n_val else None

    #pinning host memory lets the GPU copy batches over faster; only useful on cuda
    pin = cfg.device != "cpu"

    train_loader = DataLoader(
        ChunkDataset(train_tokens, cfg.context_length),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin,
        drop_last=True,
        persistent_workers=cfg.num_workers > 0,
    )

    val_loader = None
    if val_tokens is not None and len(val_tokens) > cfg.context_length + 1:
        val_loader = DataLoader(
            ChunkDataset(val_tokens, cfg.context_length),
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=pin,
            drop_last=True,
            persistent_workers=cfg.num_workers > 0,
        )

    return train_loader, val_loader


class Block(nn.Module):
    """
    one pre-norm transformer block: attention then MLP, each wrapped in a
    residual connection. pre-norm (LayerNorm before each sub-layer) is the
    stable, standard arrangement and what fixes the original norm-free model.
    """

    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim

        self.ln1 = nn.LayerNorm(cfg.d_model)
        #fused query/key/value projection: one matmul instead of three
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)

    def attention(self, x):
        batch_size, seq_len, d_model = x.shape

        #project to q, k, v and split into heads:
        #  (B, T, 3*D) -> 3 x (B, n_heads, T, head_dim)
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        #optimised attention: this calls PyTorch's fused (flash) attention
        #kernel on the GPU and applies the causal mask internally, so a token
        #can only attend to earlier tokens.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        #recombine heads back into one d_model vector per token
        out = out.transpose(1, 2).reshape(batch_size, seq_len, d_model)
        return self.proj(out)

    def forward(self, x):
        #residual + attention over a normalised input
        x = x + self.attention(self.ln1(x))
        #residual + MLP (GELU is the standard transformer activation) over a normalised input
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class GPT(nn.Module):
    """
    the whole model: token + position embeddings, a stack of transformer blocks,
    a final norm, and an output head that scores every possible next token.

    the output head shares its weights with the token embedding (weight tying),
    a standard trick that cuts parameters and tends to help small models.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        #weight tying: the output head reuses the embedding matrix
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        #small-scale init, the same spirit as the original 0.02 / 1/sqrt(d) scheme
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets=None):
        batch_size, seq_len = x.shape

        positions = torch.arange(seq_len, device=x.device)
        #hidden state starts as token meaning + position meaning
        h = self.tok_emb(x) + self.pos_emb(positions)[None, :, :]

        for block in self.blocks:
            h = block(h)

        h = self.ln_f(h)
        logits = self.head(h)

        loss = None
        if targets is not None:
            #cross-entropy over the flattened (B*T, vocab) predictions
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss


def resolve_device(name):
    """
    turn the config's device setting into a real torch.device.
    """

    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        print("device='cuda' requested but no GPU is visible; falling back to cpu", flush=True)
        name = "cpu"
    return torch.device(name)


def make_optimizer(model, cfg):
    """
    build the optimiser. adamw is the recommended modern default; sgd is offered
    for parity with the original plain-gradient-descent setup.
    """

    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg.learning_rate)
    raise ValueError(f"unknown optimizer {cfg.optimizer!r} (use 'adamw' or 'sgd')")


def lr_for_step(step, cfg):
    """
    learning-rate schedule: linear warmup then cosine decay to min_lr.

    step is 1-based. returns the LR to use for this step.
    """

    #linear warmup from 0 -> learning_rate
    if cfg.warmup_steps and step <= cfg.warmup_steps:
        return cfg.learning_rate * step / cfg.warmup_steps

    #constant LR if cosine decay is disabled
    if not cfg.lr_decay_steps:
        return cfg.learning_rate

    #cosine decay from learning_rate -> min_lr over lr_decay_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.lr_decay_steps - cfg.warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches):
    """
    average the loss over a few validation batches (text never trained on).
    """

    model.eval()
    losses = []
    for i, (x, y) in enumerate(val_loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()

    return sum(losses) / len(losses) if losses else float("nan")


@torch.no_grad()
def generate(
    prompt,
    tokenizer,
    model,
    cfg,
    max_new_tokens=100,
    temperature=0.8,
    top_k=40,
    top_p=0.95,
    device=None,
):
    """
    generate text after training, with proper sampling controls:

      temperature - <1 sharpens (more confident), >1 flattens (more random).
                    set to 0 for greedy (always the most likely token).
      top_k       - only sample from the k most likely tokens (0 = disabled).
      top_p       - nucleus sampling: keep the smallest set of tokens whose
                    probability sums to top_p (0 = disabled).
    """

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    ids = tokenizer.encode(prompt)

    for _ in range(max_new_tokens):
        #only feed the latest context_length tokens
        recent_ids = ids[-cfg.context_length:]
        x = torch.tensor(recent_ids, dtype=torch.long, device=device)[None, :]

        logits, _ = model(x)
        #use only the final position to pick the next token
        next_logits = logits[0, -1]

        #greedy when temperature is 0
        if not temperature:
            next_id = int(torch.argmax(next_logits))
            ids.append(next_id)
            continue

        next_logits = next_logits / temperature

        #top-k: keep only the k highest logits
        if top_k:
            k = min(top_k, next_logits.size(-1))
            kth_value = torch.topk(next_logits, k)[0][-1]
            next_logits = next_logits.masked_fill(next_logits < kth_value, float("-inf"))

        probs = F.softmax(next_logits, dim=-1)

        #top-p / nucleus: drop the long tail beyond cumulative prob top_p
        if top_p and top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            #mask tokens once the running total has passed top_p (keep at least one)
            remove = cumulative - sorted_probs > top_p
            sorted_probs[remove] = 0.0
            probs = torch.zeros_like(probs).scatter(0, sorted_idx, sorted_probs)
            probs = probs / probs.sum()

        next_id = int(torch.multinomial(probs, num_samples=1))
        ids.append(next_id)

    return tokenizer.decode(ids)


#config fields that describe the model architecture / data and must be saved
#alongside the weights so the model can be rebuilt for resuming or inference.
#run controls (target_loss, device, etc.) are not part of the model, so they
#are intentionally left out.
_SAVED_CONFIG_FIELDS = (
    "context_length",
    "d_model",
    "n_layers",
    "head_dim",
    "mlp_multiplier",
    "tokenpath",
    "textsource",
    "batch_size",
    "learning_rate",
    "vocab_size",
)


def save_model(path, model, optimizer, cfg, step):
    """
    save weights + optimiser state + architecture config to a single .pt file.

    the config travels with the weights so inference/resume never has to guess
    the shapes the weights were trained with; the optimiser state makes resuming
    pick up exactly where it left off (momentum and all).
    """

    config_meta = {field: getattr(cfg, field) for field in _SAVED_CONFIG_FIELDS}

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "config": config_meta,
            "step": step,
        },
        path,
    )
    print(f"model saved to {path} (step {step})", flush=True)


def load_model(path, device="cpu"):
    """
    load a model previously written by save_model.

    returns (model, cfg) where model is a GPT on `device` in eval mode and cfg
    is a Config rebuilt from the saved architecture fields. run controls are left
    at their defaults for the caller to set. used by both resume and chat.py.
    """

    device = resolve_device(device) if isinstance(device, str) else device
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config_meta = checkpoint["config"]

    cfg = Config(
        context_length=config_meta["context_length"],
        d_model=config_meta["d_model"],
        n_layers=config_meta["n_layers"],
        head_dim=config_meta["head_dim"],
        mlp_multiplier=config_meta["mlp_multiplier"],
        tokenpath=config_meta["tokenpath"],
        textsource=config_meta["textsource"],
        batch_size=config_meta["batch_size"],
        learning_rate=config_meta["learning_rate"],
        log_every=1,
        vocab_size=config_meta["vocab_size"],
    )

    model = GPT(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    return model, cfg


def train(cfg):
    """
    main training loop:
        build tokenizer + encode the corpus
        build DataLoaders (train + held-out validation)
        build model / optimiser (fresh or resumed)
        for each batch: forward -> loss -> backward -> clip -> step (under AMP)
        log, validate, checkpoint, and stop on the loss target or ctrl-c

    cfg: a fully-populated Config with all paths/hyperparameters set by the
         caller. main.py is the configuration gateway that builds it.
    """

    device = resolve_device(cfg.device)
    #store the resolved name back so the DataLoader's pin_memory decision is right
    cfg.device = device.type

    tokenizer = build_tokenizer_from_jsonl(cfg.tokenpath)

    #vocab_size must match the tokenizer because it controls:
        #token embedding rows
        #output head columns
        #number of possible next-token choices
    cfg.vocab_size = tokenizer.vocab_size

    print("Minimal hard-coded inputs:")
    print(f"vocab_size     = {cfg.vocab_size}")
    print(f"context_length = {cfg.context_length}")
    print(f"d_model        = {cfg.d_model}")
    print(f"n_layers       = {cfg.n_layers}")

    print("\nSoft-coded from those:")
    print(f"embedding_size = {cfg.vocab_size} × {cfg.d_model}")
    print(f"output_head    = {cfg.d_model} × {cfg.vocab_size} (tied to embedding)")
    print(f"head_dim       = {cfg.head_dim}")
    print(f"n_heads        = {cfg.n_heads}")
    print(f"d_ff           = {cfg.d_ff}")
    print(f"mlp_size       = {cfg.d_model} × {cfg.d_ff}")
    print(f"total_blocks   = {cfg.n_layers}")
    print(f"\ndevice         = {device} | optimizer = {cfg.optimizer} | amp = {cfg.use_amp}")

    #encode the whole corpus once, then split + wrap in DataLoaders
    tokens = encode_corpus(cfg.textsource, tokenizer)
    print(f"corpus tokens  = {len(tokens)}", flush=True)
    train_loader, val_loader = make_loaders(tokens, cfg)

    #build the model, then either resume weights or keep the fresh init
    model = GPT(cfg).to(device)
    optimizer = make_optimizer(model, cfg)
    start_step = 0

    if cfg.resume_from:
        print(f"\nresuming from {cfg.resume_from}", flush=True)
        checkpoint = torch.load(cfg.resume_from, map_location=device, weights_only=False)

        #the loaded weights only make sense if the vocabulary matches, otherwise
        #the embedding table and output head have the wrong number of rows/columns.
        saved_vocab = checkpoint["config"]["vocab_size"]
        if saved_vocab != cfg.vocab_size:
            raise ValueError(
                f"saved model vocab_size ({saved_vocab}) does not match "
                f"current tokenizer vocab_size ({cfg.vocab_size}). "
                "use the same tokenpath the model was trained with."
            )

        model.load_state_dict(checkpoint["model"])
        if checkpoint.get("optimizer") is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint.get("step", 0)

    model.train()

    #mixed precision: only meaningful on cuda. the GradScaler keeps fp16
    #gradients from underflowing to zero.
    amp_enabled = bool(cfg.use_amp) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    #wall-clock so the progress log can show how long each step takes
    last_log_time = time.time()

    #rolling window of recent losses for the early-stop target
    from collections import deque
    recent_losses = deque(maxlen=cfg.target_window)

    step = start_step

    #wrap the loop so an interruption still saves progress instead of losing it.
    try:
        #loop over the data forever: each pass over train_loader is one epoch,
        #and we keep going until the loss target is hit or ctrl-c.
        while True:
            for x, y in train_loader:
                step += 1

                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                #apply this step's scheduled learning rate
                lr = lr_for_step(step, cfg)
                for group in optimizer.param_groups:
                    group["lr"] = lr

                optimizer.zero_grad(set_to_none=True)

                #forward + loss under autocast (mixed precision on GPU)
                with torch.autocast(device_type=device.type, enabled=amp_enabled):
                    _, loss = model(x, y)

                #backward with gradient scaling for fp16 stability
                scaler.scale(loss).backward()

                #gradient clipping: unscale first so the norm is the real one
                if cfg.grad_clip:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

                scaler.step(optimizer)
                scaler.update()

                loss_value = loss.item()
                recent_losses.append(loss_value)

                #print step 1 (the baseline) then every log_every steps
                if step == start_step + 1 or step % cfg.log_every == 0:
                    now = time.time()
                    steps_since = max(1, step - (start_step if step == start_step + 1 else step - cfg.log_every))
                    per_step = (now - last_log_time) / steps_since
                    last_log_time = now
                    avg = sum(recent_losses) / len(recent_losses)
                    print(
                        f"step={step} loss={loss_value:.4f} "
                        f"avg{len(recent_losses)}={avg:.4f} "
                        f"lr={lr:.2e} ({per_step:.3f}s/step)",
                        flush=True,
                    )

                #periodic validation pass on held-out text
                if val_loader is not None and cfg.val_every and step % cfg.val_every == 0:
                    val_loss = evaluate(model, val_loader, device, cfg.val_batches)
                    print(f"  [val] step={step} val_loss={val_loss:.4f}", flush=True)

                #periodic checkpoint so progress survives a crash/interruption
                if cfg.model_output and step % cfg.checkpoint_every == 0:
                    save_model(cfg.model_output, model, optimizer, cfg, step)

                #early stop once a full window of recent losses averages to target
                if (
                    cfg.target_loss
                    and len(recent_losses) == cfg.target_window
                    and sum(recent_losses) / cfg.target_window <= cfg.target_loss
                ):
                    avg = sum(recent_losses) / cfg.target_window
                    print(
                        f"average loss over last {cfg.target_window} steps "
                        f"({avg:.4f}) reached target {cfg.target_loss} at step "
                        f"{step}, stopping",
                        flush=True,
                    )
                    raise StopIteration
    except StopIteration:
        pass
    except KeyboardInterrupt:
        #ctrl-c: stop training but keep what we have
        print(f"\ninterrupted at step {step}", flush=True)

    #persist the final weights so they can be reused for inference or resumed.
    if cfg.model_output:
        save_model(cfg.model_output, model, optimizer, cfg, step)

    print(generate("", tokenizer, model, cfg))


if __name__ == "__main__":
    #training is configured and launched from main.py, the configuration gateway
    raise SystemExit("run main.py to configure and start training")
