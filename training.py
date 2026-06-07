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
from bpe import BPETokenizer


@dataclass
class Config:
    #all of these values are configured in main.py, the configuration gateway
    #they are required arguments here so training.py holds no hardcoded settings

    #=== model architecture ===

    #number of tokens per training chunk
    context_length: int

    #per-layer hidden widths: one transformer block is built per entry, at that
    #width. this single list is the source of truth for both model depth (how
    #many blocks) and width (how wide each block is). the model starts at
    #layer_widths[0]; a projection between consecutive blocks changes the width
    #from one entry to the next. e.g. [32, 64, 128, 256] builds four blocks that
    #expand 32 -> 64 -> 128 -> 256 as data flows deeper through the stack.
    #n_layers is derived from this list, never configured separately.
    layer_widths: list[int]

    #width of one attention head. each block derives its own head count from its
    #own width: n_heads = width / head_dim. kept global (not per-layer) so the
    #attention shape logic and the RoPE cache stay simple - every head is the
    #same size regardless of which block it lives in.
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

    #where to write the trained model. this is the save folder, and its name is
    #the "save name": every checkpoint is written as its own numbered file inside
    #it, so model_output='.../model' saves into '.../model/' as model_0000001.pt,
    #model_0000002.pt, ... - one new file per save, nothing ever overwritten.
    #the model is saved on each periodic checkpoint and once more when the loss
    #target is reached; it is NOT saved if training is interrupted (ctrl-c).
    #set to a path like '.../model' to save; leave as "" / False to skip saving.
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

    #save a checkpoint every this many steps, so progress survives a crash even
    #between the start and the target. each checkpoint is a new numbered file in
    #the save folder (see model_output), so none of the history is overwritten.
    checkpoint_every: int = 10000

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

    #=== architecture options ===

    #optional path to save the encoded corpus (a .pt file holding the corpus as
    #one integer token-ID tensor, NOT the vocabulary jsonl). the first run writes
    #it; later runs load it instantly instead of re-tokenising. "" = off.
    corpus_cache: str = ""

    #use rotary position encoding (RoPE) instead of a learned position-embedding
    #table. RoPE rotates the query/key vectors by an amount that depends on each
    #token's position, so position is baked into attention itself. requires an
    #even head_dim. when True the learned pos_emb table is unused.
    use_rope: bool = False

    #use a SwiGLU gated MLP instead of the plain GELU MLP in each block. it's the
    #modern standard (Llama/Mistral) and tends to give a small quality bump; the
    #cost is one extra projection per block (slightly more params/compute).
    use_swiglu: bool = False

    #tie the output head's weights to the token embedding (a standard small-model
    #trick that saves parameters). only valid when input_width == final_width,
    #because the embedding is sized to the first width and the head to the last.
    #defaults off because the expanding-width architecture usually has different
    #first and last widths (e.g. 32 in, 256 out), which makes tying impossible.
    tie_weights: bool = False

    #number of possible token IDs
        #this gets set after the tokenizer builds the vocabulary from tokenpath
        #+ 1 padding token
        #+ 1 end-of-text token
        #+ however many token strings are found in tokenpath
    #the only field not set by main: it is filled in by train() after the tokenizer is built
    vocab_size: int = 0

    @property
    def n_layers(self):
        """number of transformer blocks - one per configured width."""

        return len(self.layer_widths)

    @property
    def input_width(self):
        """width the model starts at (token embedding + first block)."""

        return self.layer_widths[0]

    @property
    def final_width(self):
        """width the model ends at (final norm + output head)."""

        return self.layer_widths[-1]

    def n_heads_for_width(self, width):
        """attention head count for a block of the given width."""

        assert width % self.head_dim == 0
        return width // self.head_dim

    def d_ff_for_width(self, width):
        """feed-forward / MLP hidden width for a block of the given width."""

        return int(width * self.mlp_multiplier)


def validate_architecture(cfg):
    """
    fail fast, with a clear message, if a config describes an architecture the
    model cannot build. called at the top of GPT.__init__ so an invalid shape
    surfaces here rather than as a cryptic error deep inside a Linear/reshape.
    """

    if not cfg.layer_widths:
        raise ValueError("layer_widths must contain at least one width")

    for width in cfg.layer_widths:
        if width <= 0:
            raise ValueError("all layer widths must be positive")

        #each block splits its width into head_dim-sized heads, so the width has
        #to divide evenly or the attention reshape would lose/forge elements.
        if width % cfg.head_dim != 0:
            raise ValueError(
                f"width {width} must be divisible by head_dim {cfg.head_dim}"
            )

    #RoPE rotates dimensions in pairs (the two halves of each head), so a head
    #has to have an even number of dimensions.
    if cfg.use_rope and cfg.head_dim % 2 != 0:
        raise ValueError("RoPE requires an even head_dim")

    #the embedding is sized to the first width and the head to the last, so the
    #two can only share a weight matrix when those widths are equal.
    if cfg.tie_weights and cfg.input_width != cfg.final_width:
        raise ValueError(
            "tie_weights requires first and final layer widths to match "
            f"(got input_width {cfg.input_width}, final_width {cfg.final_width})"
        )


#paragraph tokenising is pure-Python and CPU-bound, so it parallelises well
#across processes. each worker gets its own copy of the tokenizer once (via the
#pool initialiser) instead of having it pickled on every task.
_ENCODE_TOKENIZER = None


def _init_encode_worker(tokenizer):
    global _ENCODE_TOKENIZER
    _ENCODE_TOKENIZER = tokenizer


def _encode_paragraph_batch(paragraphs):
    """encode a batch of paragraphs in a worker; returns one flat list of IDs."""
    tokenizer = _ENCODE_TOKENIZER
    ids = []
    for paragraph in paragraphs:
        ids.extend(tokenizer.encode(paragraph))
    return ids


def _batched(iterable, size):
    """yield successive lists of up to `size` items from an iterable."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def encode_corpus(textsource, tokenizer, cache_path="", num_workers=0):
    """
    read the whole corpus once and flatten it into a single 1-D tensor of token
    IDs (paragraphs separated by the EOS that encode() appends).

    this whole tensor is held in memory for the duration of training - the
    Dataset indexes into it to pull random chunks - so doing it once up front
    (instead of re-tokenising while training) lets the DataLoader hand out chunks
    cheaply from many worker processes at once.

    when num_workers > 0 the tokenising itself is spread across that many
    processes (paragraphs are encoded in parallel, then concatenated in order).
    if cache_path is set, that same in-memory tensor is also written to disk so
    later runs load it instantly instead of re-tokenising. progress is printed as
    it goes so a long tokenise never looks frozen.
    """

    #reuse a previously saved encoded corpus if one exists
    if cache_path and os.path.exists(cache_path):
        print(f"loading encoded corpus from {cache_path}", flush=True)
        return torch.load(cache_path)

    #how many paragraphs each task carries - big enough to amortise the
    #inter-process hand-off, small enough to keep all workers fed.
    batch_size = 2000

    start = time.time()
    all_ids = []

    if num_workers and num_workers > 0:
        from multiprocessing import Pool

        print(f"tokenising corpus (one-time) across {num_workers} processes... ", flush=True)
        approx_paragraphs = 0
        #imap keeps the results in submission order, so the corpus stays intact.
        #the pool prefetches lazily, so memory use stays bounded as we stream.
        with Pool(
            processes=num_workers,
            initializer=_init_encode_worker,
            initargs=(tokenizer,),
        ) as pool:
            batches = _batched(read_paragraphs(textsource), batch_size)
            for batch_index, ids in enumerate(pool.imap(_encode_paragraph_batch, batches), start=1):
                all_ids.extend(ids)
                approx_paragraphs = batch_index * batch_size
                #heartbeat roughly every 50k paragraphs
                if batch_index % 25 == 0:
                    print(
                        f"  ~{approx_paragraphs} paragraphs -> {len(all_ids)} token IDs "
                        f"({time.time() - start:.1f}s)",
                        flush=True,
                    )
    else:
        print("tokenising corpus (one-time)... ", flush=True)
        for n_paragraphs, paragraph in enumerate(read_paragraphs(textsource), start=1):
            all_ids.extend(tokenizer.encode(paragraph))

            #periodic heartbeat so a long tokenise is visibly progressing
            if n_paragraphs % 1000 == 0:
                print(
                    f"  {n_paragraphs} paragraphs -> {len(all_ids)} token IDs "
                    f"({time.time() - start:.1f}s)",
                    flush=True,
                )

    if len(all_ids) < 2:
        raise ValueError(
            f"corpus at {textsource!r} produced too few tokens ({len(all_ids)}) "
            "to train on. check the textsource path and the tokenizer."
        )

    #int32 is plenty for vocab sizes here and halves the memory of int64
    tokens = torch.tensor(all_ids, dtype=torch.int32)
    print(f"tokenised {len(tokens)} token IDs in {time.time() - start:.1f}s", flush=True)

    #save the encoded corpus so the next run skips tokenising entirely
    if cache_path:
        directory = os.path.dirname(cache_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        torch.save(tokens, cache_path)
        print(f"saved encoded corpus to {cache_path}", flush=True)

    return tokens


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


def build_rope_cache(seq_len, head_dim, device, base=10000.0):
    """
    precompute the cos/sin tables RoPE rotates query/key vectors by.

    returns two (seq_len, head_dim) tensors. each position gets a different set
    of rotation angles, and the rotation amount grows with position, which is
    what lets attention sense how far apart two tokens are.
    """

    #one frequency per pair of dimensions; low dims rotate fast, high dims slow
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    positions = torch.arange(seq_len, device=device).float()
    #angle for every (position, frequency) pair, then duplicated to cover both
    #halves of each dimension pair
    freqs = torch.outer(positions, inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()


def rotate_half(x):
    """
    rotate the two halves of the last dimension: (a, b) -> (-b, a). this is the
    paired-dimension trick that makes the RoPE rotation a single elementwise op.
    """

    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q, k, cos, sin):
    """
    apply the rotary position encoding to queries and keys.

    q, k:    (batch, n_heads, seq_len, head_dim)
    cos/sin: (seq_len, head_dim) -> broadcast over batch and heads
    """

    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


class Block(nn.Module):
    """
    one pre-norm transformer block: attention then MLP, each wrapped in a
    residual connection. pre-norm (LayerNorm before each sub-layer) is the
    stable, standard arrangement and what fixes the original norm-free model.
    """

    def __init__(self, width, head_dim, mlp_multiplier, use_swiglu):
        super().__init__()
        #the block is fixed-width internally: it takes a `width`-vector per token
        #and returns a `width`-vector, so attention/MLP outputs can be added back
        #onto the residual stream. width changes happen between blocks, not here.
        self.width = width
        self.head_dim = head_dim
        self.n_heads = width // head_dim
        self.d_ff = int(width * mlp_multiplier)
        self.use_swiglu = use_swiglu

        self.ln1 = nn.LayerNorm(width)
        #fused query/key/value projection: one matmul instead of three
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

        self.ln2 = nn.LayerNorm(width)
        if use_swiglu:
            #SwiGLU: two input projections (a "gate" and an "up"), combined as
            #silu(gate) * up, then projected back down. the gate lets the block
            #learn which features to let through, which plain GELU can't.
            self.fc_gate = nn.Linear(width, self.d_ff)
            self.fc_up = nn.Linear(width, self.d_ff)
            self.fc_down = nn.Linear(self.d_ff, width)
        else:
            #plain MLP: expand, GELU, shrink back
            self.fc1 = nn.Linear(width, self.d_ff)
            self.fc2 = nn.Linear(self.d_ff, width)

    def mlp(self, x):
        if self.use_swiglu:
            return self.fc_down(F.silu(self.fc_gate(x)) * self.fc_up(x))
        return self.fc2(F.gelu(self.fc1(x)))

    def attention(self, x, rope):
        batch_size, seq_len, d_model = x.shape

        #project to q, k, v and split into heads:
        #  (B, T, 3*D) -> 3 x (B, n_heads, T, head_dim)
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        #rotary position encoding: rotate q/k by their position before attention.
        #rope is (cos, sin) when enabled, or None when using learned pos_emb.
        if rope is not None:
            cos, sin = rope
            q, k = apply_rope(q, k, cos, sin)

        #optimised attention: this calls PyTorch's fused (flash) attention
        #kernel on the GPU and applies the causal mask internally, so a token
        #can only attend to earlier tokens.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        #recombine heads back into one d_model vector per token
        out = out.transpose(1, 2).reshape(batch_size, seq_len, d_model)
        return self.proj(out)

    def forward(self, x, rope):
        #residual + attention over a normalised input
        x = x + self.attention(self.ln1(x), rope)
        #residual + MLP (SwiGLU or plain GELU) over a normalised input
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """
    the whole model: token embeddings (+ position info), a stack of transformer
    blocks at progressively-changing widths, a final norm, and an output head
    that scores every possible next token.

    the stack is variable-width: each block is built at its own width from
    cfg.layer_widths, and a linear projection between consecutive blocks carries
    the hidden state from one width to the next. the model starts at input_width
    (the first entry) and ends at final_width (the last), so for
    [32, 64, 128, 256] the hidden state grows 32 -> 64 -> 128 -> 256 with depth.

    position is handled one of two ways depending on cfg.use_rope:
      - RoPE: queries/keys are rotated by position inside attention (no table)
      - learned: a position-embedding table is added to the token vectors

    the output head can optionally share its weights with the token embedding
    (cfg.tie_weights), but only when input_width == final_width; the expanding
    architecture usually has different first/last widths, so tying is off by
    default.
    """

    def __init__(self, cfg):
        super().__init__()
        #reject invalid architectures up front with a clear message
        validate_architecture(cfg)

        self.cfg = cfg
        self.use_rope = cfg.use_rope

        #the model begins at the first configured width: token (and position)
        #vectors are input_width-wide before they reach the first block.
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.input_width)

        if self.use_rope:
            #precompute the rotation tables once and ship them with the model
            #(non-persistent: rebuilt on load, never saved into the checkpoint).
            #head_dim is global, so one cache serves every block's attention.
            cos, sin = build_rope_cache(cfg.context_length, cfg.head_dim, device="cpu")
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            self.pos_emb = nn.Embedding(cfg.context_length, cfg.input_width)

        #one transformer block per configured width, plus one projection between
        #each pair of consecutive widths to carry the hidden state across the
        #width change. for [32, 64, 128, 256] that's 4 blocks and 3 projections:
        #  Block(32) -> Linear(32->64) -> Block(64) -> Linear(64->128) -> ...
        self.blocks = nn.ModuleList()
        self.width_projections = nn.ModuleList()
        for i, width in enumerate(cfg.layer_widths):
            self.blocks.append(
                Block(
                    width=width,
                    head_dim=cfg.head_dim,
                    mlp_multiplier=cfg.mlp_multiplier,
                    use_swiglu=cfg.use_swiglu,
                )
            )
            if i < len(cfg.layer_widths) - 1:
                next_width = cfg.layer_widths[i + 1]
                self.width_projections.append(nn.Linear(width, next_width))

        #final norm + output head operate at the last (widest) width
        self.ln_f = nn.LayerNorm(cfg.final_width)
        self.head = nn.Linear(cfg.final_width, cfg.vocab_size, bias=False)

        #weight tying is optional and only possible when the embedding and head
        #are the same width. validate_architecture already enforced this, but the
        #guard keeps the tying self-documenting and safe if called directly.
        if cfg.tie_weights:
            if cfg.input_width != cfg.final_width:
                raise ValueError("tie_weights requires input_width == final_width")
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

        if self.use_rope:
            #position lives in the rotation applied inside attention; the hidden
            #state starts as token meaning only. slice the cached tables to this
            #sequence length (and follow the model's device/dtype).
            h = self.tok_emb(x)
            rope = (
                self.rope_cos[:seq_len].to(h.dtype),
                self.rope_sin[:seq_len].to(h.dtype),
            )
        else:
            positions = torch.arange(seq_len, device=x.device)
            #hidden state starts as token meaning + position meaning
            h = self.tok_emb(x) + self.pos_emb(positions)[None, :, :]
            rope = None

        #run each block at its own width, then project the hidden state up to the
        #next block's width. the final block has no projection after it - its
        #output flows straight into the final norm and output head.
        for i, block in enumerate(self.blocks):
            h = block(h, rope)
            if i < len(self.width_projections):
                h = self.width_projections[i](h)

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
    repetition_penalty=1.1,
    device=None,
):
    """
    generate text after training, with proper sampling controls:

      temperature        - <1 sharpens (more confident), >1 flattens (more
                           random). set to 0 for greedy (always the most likely
                           token).
      top_k              - only sample from the k most likely tokens (0 = off).
      top_p              - nucleus sampling: keep the smallest set of tokens
                           whose probability sums to top_p (0 = off).
      repetition_penalty - >1 discourages repeating tokens already generated,
                           which stops small models looping ("the the the").
                           1.0 = off; ~1.1-1.3 is a typical range.
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

        #repetition penalty: push down the score of tokens we've already emitted.
        #divide positive logits / multiply negative ones so the penalty always
        #moves a token toward "less likely" regardless of its sign.
        if repetition_penalty and repetition_penalty != 1.0:
            for token_id in set(ids):
                if next_logits[token_id] > 0:
                    next_logits[token_id] /= repetition_penalty
                else:
                    next_logits[token_id] *= repetition_penalty

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
    "layer_widths",
    "head_dim",
    "mlp_multiplier",
    "tokenpath",
    "textsource",
    "batch_size",
    "learning_rate",
    "vocab_size",
    "use_rope",
    "use_swiglu",
    "tie_weights",
)


def _architecture_from_meta(config_meta):
    """
    reconstruct the architecture-defining fields (layer_widths, tie_weights) from
    a saved checkpoint's config, transparently upgrading older single-width
    checkpoints that predate these fields.

    old checkpoints stored a single d_model + n_layers and tied the output head
    unconditionally; that is equivalent to a uniform-width stack with tying on
    (its first and last widths are equal, so tying was always valid). newer
    checkpoints store layer_widths and tie_weights directly.

    returns (layer_widths, tie_weights).
    """

    if "layer_widths" in config_meta:
        layer_widths = list(config_meta["layer_widths"])
    else:
        #a same-width stack: d_model repeated n_layers times
        layer_widths = [config_meta["d_model"]] * config_meta["n_layers"]

    #new checkpoints carry the real flag; for old ones, mirror the unconditional
    #tying the original file did (only ever valid because the widths were uniform)
    tie_weights = config_meta.get("tie_weights", layer_widths[0] == layer_widths[-1])

    return layer_widths, tie_weights


def _check_resume_compatible(saved_config, cfg):
    """
    raise a clear, specific error if a checkpoint cannot be resumed into the
    current config, instead of letting load_state_dict fail with a cryptic shape
    mismatch. resume is intentionally strict: any architecture change starts a
    fresh model rather than trying to partially load the old weights.
    """

    #the loaded weights only make sense if the vocabulary matches, otherwise the
    #embedding table and output head have the wrong number of rows/columns.
    saved_vocab = saved_config["vocab_size"]
    if saved_vocab != cfg.vocab_size:
        raise ValueError(
            f"saved model vocab_size ({saved_vocab}) does not match "
            f"current tokenizer vocab_size ({cfg.vocab_size}). "
            "use the same tokenpath the model was trained with."
        )

    saved_layer_widths, saved_tie_weights = _architecture_from_meta(saved_config)

    #every field below changes the parameter shapes or the forward behaviour, so
    #a mismatch must stop the resume rather than load into the wrong architecture.
    if saved_layer_widths != cfg.layer_widths:
        raise ValueError(
            f"saved model layer_widths {saved_layer_widths} do not match "
            f"current layer_widths {cfg.layer_widths}"
        )

    if saved_config["head_dim"] != cfg.head_dim:
        raise ValueError(
            f"saved model head_dim {saved_config['head_dim']} does not match "
            f"current head_dim {cfg.head_dim}"
        )

    if saved_config["mlp_multiplier"] != cfg.mlp_multiplier:
        raise ValueError(
            f"saved model mlp_multiplier {saved_config['mlp_multiplier']} does "
            f"not match current mlp_multiplier {cfg.mlp_multiplier}"
        )

    saved_use_rope = saved_config.get("use_rope", False)
    if saved_use_rope != cfg.use_rope:
        raise ValueError(
            f"saved model use_rope {saved_use_rope} does not match "
            f"current use_rope {cfg.use_rope}"
        )

    saved_use_swiglu = saved_config.get("use_swiglu", False)
    if saved_use_swiglu != cfg.use_swiglu:
        raise ValueError(
            f"saved model use_swiglu {saved_use_swiglu} does not match "
            f"current use_swiglu {cfg.use_swiglu}"
        )

    if saved_tie_weights != cfg.tie_weights:
        raise ValueError(
            f"saved model tie_weights {saved_tie_weights} does not match "
            f"current tie_weights {cfg.tie_weights}"
        )


#how many digits each checkpoint number is zero-padded to in its filename. a
#generous width keeps a plain alphabetical listing of the save folder in the
#same order the checkpoints were written.
_CHECKPOINT_PAD = 7


def checkpoint_folder(model_output):
    """
    the folder every numbered checkpoint for a given save name is written into.

    the folder is named after the save name (the model_output stem), so
    model_output='/a/b/model.pt' -> '/a/b/model'.
    """

    directory = os.path.dirname(model_output)
    stem = os.path.splitext(os.path.basename(model_output))[0]
    return os.path.join(directory, stem) if directory else stem


def checkpoint_path(model_output, number):
    """
    build the path for one numbered checkpoint inside the save folder.

    e.g. model_output='/a/b/model.pt', number=1 -> '/a/b/model/model_0000001.pt'.
    every save is its own file so the folder keeps the whole history instead of a
    single overwritten model; the number is zero-padded so the files sort in order.
    """

    stem = os.path.splitext(os.path.basename(model_output))[0]
    filename = f"{stem}_{number:0{_CHECKPOINT_PAD}d}.pt"
    return os.path.join(checkpoint_folder(model_output), filename)


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
            #record which optimizer made this state so a resume can tell whether
            #the saved state is compatible (adamw state can't load into sgd, etc.)
            "optimizer_type": cfg.optimizer,
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

    #rebuild the architecture, upgrading older single-width checkpoints on the fly
    layer_widths, tie_weights = _architecture_from_meta(config_meta)

    cfg = Config(
        context_length=config_meta["context_length"],
        layer_widths=layer_widths,
        head_dim=config_meta["head_dim"],
        mlp_multiplier=config_meta["mlp_multiplier"],
        tokenpath=config_meta["tokenpath"],
        textsource=config_meta["textsource"],
        batch_size=config_meta["batch_size"],
        learning_rate=config_meta["learning_rate"],
        log_every=1,
        #.get(...) defaults keep models saved before these options existed loadable
        use_rope=config_meta.get("use_rope", False),
        use_swiglu=config_meta.get("use_swiglu", False),
        tie_weights=tie_weights,
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

    tokenizer = BPETokenizer.load(cfg.tokenpath)

    #vocab_size must match the tokenizer because it controls:
        #token embedding rows
        #output head columns
        #number of possible next-token choices
    cfg.vocab_size = tokenizer.vocab_size

    #per-block attention head counts and MLP hidden widths, derived from each
    #block's width - makes it obvious the intended variable-width stack is built.
    heads_by_layer = [cfg.n_heads_for_width(w) for w in cfg.layer_widths]
    d_ff_by_layer = [cfg.d_ff_for_width(w) for w in cfg.layer_widths]

    print("Minimal hard-coded inputs:")
    print(f"vocab_size     = {cfg.vocab_size}")
    print(f"context_length = {cfg.context_length}")
    print(f"layer_widths   = {cfg.layer_widths}")
    print(f"n_layers       = {cfg.n_layers}")

    print("\nSoft-coded from those:")
    print(f"input_width    = {cfg.input_width}")
    print(f"final_width    = {cfg.final_width}")
    print(f"embedding_size = {cfg.vocab_size} × {cfg.input_width}")
    tied_note = " (tied to embedding)" if cfg.tie_weights else ""
    print(f"output_head    = {cfg.final_width} × {cfg.vocab_size}{tied_note}")
    print(f"head_dim       = {cfg.head_dim}")
    print(f"heads_by_layer = {heads_by_layer}")
    print(f"d_ff_by_layer  = {d_ff_by_layer}")
    print(f"total_blocks   = {cfg.n_layers}")
    print(f"position_enc   = {'RoPE' if cfg.use_rope else 'learned'}")
    print(f"mlp_type       = {'SwiGLU' if cfg.use_swiglu else 'GELU'}")
    print(f"tie_weights    = {cfg.tie_weights}")
    print(f"\ndevice         = {device} | optimizer = {cfg.optimizer} | amp = {cfg.use_amp}")

    #make checkpointing status obvious up front so a silent "model_output = False"
    #never looks like training that mysteriously saved nothing.
    if cfg.model_output:
        print(
            f"checkpoints    = ON -> {checkpoint_folder(cfg.model_output)}/ "
            f"(a new numbered file every {cfg.checkpoint_every} steps; "
            f"not saved on interrupt)"
        )
    else:
        print("checkpoints    = OFF (model_output is not set; nothing will be saved!)")

    #encode the whole corpus once, then split + wrap in DataLoaders
    tokens = encode_corpus(
        cfg.textsource,
        tokenizer,
        cache_path=cfg.corpus_cache,
        num_workers=cfg.num_workers,
    )
    print(f"corpus tokens  = {len(tokens)}", flush=True)
    train_loader, val_loader = make_loaders(tokens, cfg)

    #build the model, then either resume weights or keep the fresh init
    model = GPT(cfg).to(device)
    optimizer = make_optimizer(model, cfg)
    start_step = 0

    if cfg.resume_from:
        print(f"\nresuming from {cfg.resume_from}", flush=True)
        checkpoint = torch.load(cfg.resume_from, map_location=device, weights_only=False)

        #the weights only make sense if the vocabulary and architecture match;
        #this raises a clear error on any mismatch instead of a cryptic shape
        #failure inside load_state_dict. resume is strict by design.
        _check_resume_compatible(checkpoint["config"], cfg)

        model.load_state_dict(checkpoint["model"])

        #only reload optimizer state if it came from the same optimizer type.
        #adamw state (momentum + variance) is incompatible with sgd and vice
        #versa, so on a mismatch we keep the model weights but start the
        #optimizer fresh instead of crashing.
        saved_optimizer = checkpoint.get("optimizer_type")
        if checkpoint.get("optimizer") is not None and saved_optimizer == cfg.optimizer:
            optimizer.load_state_dict(checkpoint["optimizer"])
        elif checkpoint.get("optimizer") is not None:
            print(
                f"optimizer changed ({saved_optimizer} -> {cfg.optimizer}); "
                "keeping model weights but starting optimizer state fresh",
                flush=True,
            )
        start_step = checkpoint.get("step", 0)

    model.train()

    #mixed precision: only meaningful on cuda. the GradScaler keeps fp16
    #gradients from underflowing to zero.
    amp_enabled = bool(cfg.use_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

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

                #periodic checkpoint so progress survives a crash; each save is a
                #new numbered file (the Nth checkpoint) inside the save folder.
                if cfg.model_output and step % cfg.checkpoint_every == 0:
                    number = step // cfg.checkpoint_every
                    save_model(
                        checkpoint_path(cfg.model_output, number),
                        model, optimizer, cfg, step,
                    )

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
        #target reached: capture the finished model. if this step already landed
        #on a checkpoint boundary it was just saved in the loop, so only add a
        #final file when the last checkpoint isn't already this exact step. it is
        #numbered just past the last periodic checkpoint so it sorts last.
        if cfg.model_output and step % cfg.checkpoint_every != 0:
            number = step // cfg.checkpoint_every + 1
            save_model(
                checkpoint_path(cfg.model_output, number),
                model, optimizer, cfg, step,
            )
    except KeyboardInterrupt:
        #ctrl-c: stop training but keep what we have. by design we do NOT save on
        #interrupt - only the periodic checkpoints already on disk are kept.
        print(f"\ninterrupted at step {step} (not saving on interrupt)", flush=True)

    print(generate("", tokenizer, model, cfg))


if __name__ == "__main__":
    #training is configured and launched from main.py, the configuration gateway
    raise SystemExit("run main.py to configure and start training")
