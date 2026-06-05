"""Model backend: HuggingFace/PyTorch loader and forward-hook capture.

The plan's firing-behavior core requires forward hooks on residual streams, MLP
pre/post activations, and attention internals. The Ollama GGUF referenced by
``/rescaling`` cannot be hooked cleanly, so the default (and currently only)
backend loads the same Llama 3 family in HuggingFace format from the shared data
directory and registers hooks on the decoder layers.

All ``torch``/``transformers`` imports are lazy so the rest of the library
(schemas, storage, IDs, sketches) can be imported and tested without a GPU or a
multi-gigabyte checkpoint present. This is important because the extraction runs
on the user's machine, not in the environment where the package is edited.

Unit conventions for a Llama-style decoder block:

  * MLP neuron activation = ``act_fn(gate_proj(x)) * up_proj(x)`` — captured as
    the *input to* ``down_proj`` via a forward pre-hook. One unit per
    intermediate dimension.
  * Attention head activation = L2 norm of each head's slice of the input to
    ``o_proj`` — captured via a forward pre-hook on ``o_proj``. One unit per head.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator


@dataclass
class Architecture:
    """Architecture metadata needed to define units and read tensors."""

    model_type: str
    num_layers: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    vocab_size: int

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


#Per-layer tensor roles for a Llama-style decoder, used to build the catalog.
LLAMA_LAYER_TENSORS = {
    "self_attn.q_proj.weight": ("attention", "q_proj"),
    "self_attn.k_proj.weight": ("attention", "k_proj"),
    "self_attn.v_proj.weight": ("attention", "v_proj"),
    "self_attn.o_proj.weight": ("attention", "o_proj"),
    "mlp.gate_proj.weight": ("mlp", "gate_proj"),
    "mlp.up_proj.weight": ("mlp", "up_proj"),
    "mlp.down_proj.weight": ("mlp", "down_proj"),
    "input_layernorm.weight": ("norm", "input_layernorm"),
    "post_attention_layernorm.weight": ("norm", "post_attention_layernorm"),
}


class ModelBackend:
    """Lazy wrapper around an HF causal-LM with hook-based activation capture."""

    def __init__(self, model_path: str, tokenizer_path: str | None = None,
                 dtype: str = "bfloat16", device: str = "auto",
                 dequantize_f16: bool = False, llama_quantize: str = "llama-quantize",
                 build_dir: str | None = None):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.dtype = dtype
        self.device = device
        self.dequantize_f16 = dequantize_f16
        self.llama_quantize = llama_quantize
        self.build_dir = build_dir
        self._model = None
        self._tokenizer = None
        self._config = None
        self._gguf = None  # resolved GGUFSource, or False if this is a HF dir

    #--- GGUF resolution --------------------------------------------------
    def _gguf_source(self):
        """Resolve (and cache) the GGUF source, or return None for HF dirs.

        Mirrors /rescaling: an Ollama model name or .gguf path is resolved to a
        local GGUF blob, optionally dequantized to F16 first. ``transformers``
        then loads it directly, dequantizing in memory — no HF download needed.
        """
        if self._gguf is None:
            from . import gguf_source
            if gguf_source.looks_like_gguf_request(self.model_path):
                src = gguf_source.resolve_gguf_source(self.model_path)
                if self.dequantize_f16:
                    build = Path(self.build_dir) if self.build_dir else src.gguf_path.parent / "build"
                    src = gguf_source.dequantize_to_f16(
                        src.gguf_path, build, self.llama_quantize)
                self._gguf = src
            else:
                self._gguf = False
        return self._gguf or None

    def _gguf_kwargs(self) -> dict:
        src = self._gguf_source()
        return {"gguf_file": src.filename} if src else {}

    def _load_dir(self) -> str:
        src = self._gguf_source()
        return src.directory if src else self.model_path

    #--- loading ----------------------------------------------------------
    def load_config_only(self):
        """Read the model config. For GGUF sources this loads the full model
        (the config is embedded in the GGUF), which is needed at init anyway."""
        if self._config is not None:
            return self._config
        if self._gguf_source() is not None:
            self.load()
            return self._config
        from transformers import AutoConfig
        self._config = AutoConfig.from_pretrained(self.model_path)
        return self._config

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        gguf_kwargs = self._gguf_kwargs()
        load_dir = self._load_dir()
        dtype = getattr(torch, self.dtype, torch.float32)
        device_map = self.device if self.device != "cpu" else None
        self._model = AutoModelForCausalLM.from_pretrained(
            load_dir, torch_dtype=dtype,
            device_map=device_map, output_hidden_states=False, **gguf_kwargs,
        )
        self._model.eval()
        tok_dir = self._load_dir() if self._gguf_source() else self.tokenizer_path
        self._tokenizer = AutoTokenizer.from_pretrained(tok_dir, **gguf_kwargs)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._config = self._model.config

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            from transformers import AutoTokenizer
            gguf_kwargs = self._gguf_kwargs()
            tok_dir = self._load_dir() if self._gguf_source() else self.tokenizer_path
            self._tokenizer = AutoTokenizer.from_pretrained(tok_dir, **gguf_kwargs)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    #--- architecture -----------------------------------------------------
    def architecture(self) -> Architecture:
        cfg = self.load_config_only()
        num_heads = getattr(cfg, "num_attention_heads")
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // num_heads
        return Architecture(
            model_type=getattr(cfg, "model_type", "unknown"),
            num_layers=getattr(cfg, "num_hidden_layers"),
            hidden_size=cfg.hidden_size,
            intermediate_size=getattr(cfg, "intermediate_size"),
            num_attention_heads=num_heads,
            num_key_value_heads=getattr(cfg, "num_key_value_heads", num_heads),
            head_dim=head_dim,
            vocab_size=getattr(cfg, "vocab_size"),
        )

    def param_count(self) -> int:
        """Total parameter count (loads weights if not already loaded)."""
        self.load()
        return sum(p.numel() for p in self._model.parameters())

    #--- tensor catalog ---------------------------------------------------
    def iter_named_tensors(self) -> Iterator[tuple[str, tuple[int, ...], str]]:
        """Yield (name, shape, dtype) for every weight tensor.

        Uses the state dict metadata; for large models prefer the safetensors
        header to avoid materializing tensors, but the simple path works for the
        catalog because we only read shapes here.
        """
        self.load()
        for name, tensor in self._model.state_dict().items():
            yield name, tuple(tensor.shape), str(tensor.dtype).replace("torch.", "")

    def get_tensor(self, name: str):
        """Return a single weight tensor as a numpy array (float32)."""
        self.load()
        sd = self._model.state_dict()
        if name not in sd:
            raise KeyError(name)
        return sd[name].to("cpu").float().numpy()

    #--- activation capture ----------------------------------------------
    def capture(self, input_ids, attention_mask, layer_callback: Callable):
        """Run a forward pass, invoking ``layer_callback`` with captured tensors.

        ``input_ids`` and ``attention_mask`` are plain NumPy arrays; this method
        owns all torch conversions and device placement so callers (and tests)
        never need torch. ``layer_callback(layer_id, captured)`` is called once
        per decoder layer, where ``captured`` is a dict with:
            "mlp"            -> np.ndarray [B, T, intermediate]  (neuron acts)
            "attn_head_norm" -> np.ndarray [B, T, num_heads]     (head output norm)

        The hooks store tensors transiently for one forward pass and are removed
        afterward, so no per-layer state leaks between batches.
        """
        import torch

        self.load()
        arch = self.architecture()
        layers = self._decoder_layers()
        captured: dict[int, dict[str, Any]] = {}
        handles = []

        input_ids = torch.as_tensor(input_ids)
        attention_mask = torch.as_tensor(attention_mask)
        try:
            dev = next(self._model.parameters()).device
            input_ids = input_ids.to(dev)
            attention_mask = attention_mask.to(dev)
        except StopIteration:
            pass

        def make_mlp_hook(idx: int):
            def hook(module, args):
                #Pre-hook on down_proj: args[0] is the intermediate activation.
                x = args[0].detach().to(torch.float32).cpu().numpy()
                captured.setdefault(idx, {})["mlp"] = x
            return hook

        def make_attn_hook(idx: int):
            def hook(module, args):
                #Pre-hook on o_proj: args[0] is [B, T, num_heads*head_dim].
                x = args[0].detach().to(torch.float32)
                b, t, _ = x.shape
                x = x.view(b, t, arch.num_attention_heads, arch.head_dim)
                norms = x.norm(dim=-1).cpu().numpy()  # [B, T, num_heads]
                captured.setdefault(idx, {})["attn_head_norm"] = norms
            return hook

        for idx, layer in enumerate(layers):
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(make_mlp_hook(idx)))
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(make_attn_hook(idx)))

        try:
            with torch.no_grad():
                self._model(input_ids=input_ids, attention_mask=attention_mask)
        finally:
            for h in handles:
                h.remove()

        for idx in range(arch.num_layers):
            if idx in captured:
                layer_callback(idx, captured[idx])

    def _decoder_layers(self):
        #Llama: model.model.layers ; fall back to common attribute paths.
        m = self._model
        for path in ("model.layers", "transformer.h", "gpt_neox.layers"):
            obj = m
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
                return obj
            except AttributeError:
                continue
        raise AttributeError("Could not locate decoder layers on the model.")
