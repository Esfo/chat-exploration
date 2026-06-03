from dataclasses import dataclass
import numpy as np
import math
import json

from read_paragraphs import read_paragraphs


@dataclass
class Config:
    #number of possible token IDs
        #this gets set after the tokenizer builds the vocabulary from tokenpath
        #+ 1 padding token
        #+ 1 end-of-text token
        #+ however many token strings are found in tokenpath
    vocab_size: int = 0

    #number of tokens per training chunk
    context_length: int = 128

    #width of the model's internal token vector
    #means every token becomes a vector with 256 numbers
    d_model: int = 256

    #number of transformer blocks
    #each block contains:
        #attention
        #feed-forward / MLP
    n_layers: int = 2

    #width of one attention head.
    #n_heads is calculated from: n_heads = d_model / head_dim
    head_dim: int = 64

    #controls how wide the MLP part gets inside each transformer block
    #(multi-layer perceptron)
    mlp_multiplier: float = 4.0

    #path to the token word list the tokenizer is built from
    tokenpath: str = '/home/sfo/data/models/tokens/text-chunks.jsonl'

    #corpus the training paragraphs are read from
    textsource: str = '/home/sfo/store/gutenberg/gutenbooks/'

    #number of chunks trained together in one update
    batch_size: int = 8

    #how large each training update is
    learning_rate: float = 1e-3

    #get batch, predict next tokens, calculate loss, calculate gradients, update weights, repeat __this-many__ times
    train_steps: int = 2000

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

    #not actively used here because chunks are always exactly context_length tokens
    #reserved so token ID 0 never means a real token
    #would be used to fill shorter examples if you later train variable-length chunks
    pad_id: int = 0

    #stop codon
    eos_id: int = 1

    #maps each token string to one token ID
    token_to_id: dict

    #maps each token ID back to one token string
    id_to_token: dict

    #tokens sorted longest-first so multi-character tokens are matched before shorter tokens
    match_tokens: list

    #number of possible token IDs
    vocab_size: int

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

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)

            if "text" not in row:
                raise ValueError("Each JSONL row must contain a 'text' field.")

            text = row["text"].strip()

            if text == "":
                continue

            #each whitespace-separated item is treated as one token string
            for token in text.split():
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


def yield_jsonl_text(textsource):
    """
    stream paragraphs from the corpus, the same way they're fed into word survival
    """

    #loop forever, yielding more paragraphs whenever the buffer needs more context
    while True:
        for paragraph in read_paragraphs(textsource):
            yield paragraph


def yield_token_chunks(textsource, tokenizer, context_length):
    """
    convert streamed text into fixed-size next-token training chunk
    """

    #rolling token storage
    #this collects tokens until we have enough to produce a training chunk
    buffer = []

    for text in yield_jsonl_text(textsource):
        #convert text row into token IDs and add it to the rolling buffer
        buffer.extend(tokenizer.encode(text))

        #while there are enough tokens to create one training example
        while len(buffer) >= context_length + 1:
            #take 129 tokens if context_length is 128
            chunk = buffer[: context_length + 1]

            #move the buffer forward by context_length
            #this creates mostly non-overlapping chunks
            buffer = buffer[context_length:]

            #x is what the model sees
            x = np.array(chunk[:-1], dtype=np.int64)

            #y is what the model should predict
            y = np.array(chunk[1:], dtype=np.int64)

            yield x, y


def batch_stream(textsource, tokenizer, context_length, batch_size):
    """
    group individual chunks into batches for the model to train on
    """

    #create a generator that yields one x/y training chunk at a time
    stream = yield_token_chunks(textsource, tokenizer, context_length)

    #keep producing batches forever
    #training stops elsewhere when train_steps is reached
    while True:
        #xs will hold input chunks
        #ys will hold target chunks
        xs = []
        ys = []

        #collect batch_size individual chunks
        for _ in range(batch_size):
            #x = input token IDs
            #y = target next-token IDs
            x, y = next(stream)

            #add this example to the batch
            xs.append(x)
            ys.append(y)

        #turn lists of arrays into one batch array
        #x shape becomes batch_size × context_length
        #y shape becomes batch_size × context_length
        yield np.stack(xs), np.stack(ys)


def softmax(x):
    """
    convert raw scores aka logits into probabilities
    """

    #subtract the largest score for numerical stability
    #this does not change the final probabilities
    #it prevents np.exp() from overflowing on very large numbers
    x = x - np.max(x, axis=-1, keepdims=True)

    #turn scores into positive numbers
    #larger logits become much larger exp values
    exp_x = np.exp(x)

    #divide each exp score by the sum of all exp scores so the final values are probabilities that add up to 1
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def cross_entropy(logits, targets):
    """
    calculate prediction loss and gradient
    """

    #convert raw output scores into probabilities
    #shape stays batch_size × seq_len × vocab_size
    probs = softmax(logits)
    
    #batch_size = number of examples trained together
    #seq_len = number of token positions per example
    #vocab_size = number of possible next-token choices
    batch_size, seq_len, vocab_size = logits.shape
    
    #flatten batch and sequence into one long list of predictions
    #before: batch_size × seq_len × vocab_size
    #after:  (batch_size * seq_len) × vocab_size
    flat_probs = probs.reshape(batch_size * seq_len, vocab_size)
    
    #flatten targets to match flat_probs
    #before: batch_size × seq_len
    #after:  batch_size * seq_len
    flat_targets = targets.reshape(batch_size * seq_len)
    
    #get the probability the model assigned to the correct token at every position
    correct_token_probs = flat_probs[np.arange(batch_size * seq_len), flat_targets]
    
    #negative log-likelihood
    #high probability for correct token -> low loss
    #low probability for correct token -> high loss
    #1e-12 prevents log(0), which would be infinite
    loss = -np.mean(np.log(correct_token_probs + 1e-12))
    
    #gradient of softmax + cross entropy:
    
    #start with probabilities
    dlogits = flat_probs.copy()
    
    #subtract 1 from the correct token position
    #this creates the direction the logits should move
    dlogits[np.arange(batch_size * seq_len), flat_targets] -= 1
    
    #average gradient over all token predictions in the batch
    dlogits /= batch_size * seq_len
    
    #reshape gradient back to original logits shape
    return loss, dlogits.reshape(batch_size, seq_len, vocab_size)


def init_params(cfg):
    """
    initialize all trainable model weights
    """

    #create a NumPy random number generator
    #use seed 1 so the random numbers are repeatable
    rng = np.random.default_rng(1)
    p = {}

    #token embedding table
    #0.02 is a common small initialization scale
    p["tok_emb"] = rng.normal(
        loc=0.0,
        scale=0.02,
        size=(cfg.vocab_size, cfg.d_model),
    )

    #position embedding table
    p["pos_emb"] = rng.normal(
        loc=0.0,
        scale=0.02,
        size=(cfg.context_length, cfg.d_model),
    )

    for layer in range(cfg.n_layers):
        #attention weights

        #Wq = query projection
        #why 1 / sqrt(d_model)?
            #keeps the starting activation sizes reasonably stable
            #if d_model is larger, each output sums more inputs, so each weight should start smaller
        p[f"{layer}.Wq"] = rng.normal(
            loc=0.0,
            scale=1 / math.sqrt(cfg.d_model),
            size=(cfg.d_model, cfg.d_model),
        )
        
        #Wk = key projection
        p[f"{layer}.Wk"] = rng.normal(
            loc=0.0,
            scale=1 / math.sqrt(cfg.d_model),
            size=(cfg.d_model, cfg.d_model),
        )

        #Wv = value projection
        p[f"{layer}.Wv"] = rng.normal(
            loc=0.0,
            scale=1 / math.sqrt(cfg.d_model),
            size=(cfg.d_model, cfg.d_model),
        )

        #Wo = output projection
        p[f"{layer}.Wo"] = rng.normal(
            loc=0.0,
            scale=1 / math.sqrt(cfg.d_model),
            size=(cfg.d_model, cfg.d_model),
        )

        #MLP first layer
        #this expands each token vector
        p[f"{layer}.W1"] = rng.normal(
            loc=0.0,
            scale=1 / math.sqrt(cfg.d_model),
            size=(cfg.d_model, cfg.d_ff),
        )

        #MLP first bias
        p[f"{layer}.b1"] = np.zeros((cfg.d_ff,))

        #MLP second layer
        #this shrinks the vector back to d_model
        p[f"{layer}.W2"] = rng.normal(
            loc=0.0,
            scale=1 / math.sqrt(cfg.d_ff),
            size=(cfg.d_ff, cfg.d_model),
        )

        #MLP second bias
        p[f"{layer}.b2"] = np.zeros((cfg.d_model,))

    #final output head
    #convert each token's internal vector into scores for every possible next token.
    p["Wout"] = rng.normal(
        loc=0.0,
        scale=1 / math.sqrt(cfg.d_model),
        size=(cfg.d_model, cfg.vocab_size),
    )

    #output bias
    p["bout"] = np.zeros((cfg.vocab_size,))

    return p


def attention_forward(h, p, layer, cfg):
    """
    run attention for one transformer block
    """

    batch_size, seq_len, d_model = h.shape

    n_heads = cfg.n_heads
    head_dim = cfg.head_dim

    #project hidden state into queries, keys, values:

    #multiply hidden states by the query weight matrix
    #this creates query vectors: what each token is looking for
    q = h @ p[f"{layer}.Wq"]
    
    #multiply hidden states by the key weight matrix
    #this creates key vectors: what each token offers for matching
    k = h @ p[f"{layer}.Wk"]
    
    #multiply hidden states by the value weight matrix
    #this creates value vectors: what information each token can pass along
    v = h @ p[f"{layer}.Wv"]

    #split d_model into multiple attention heads
        #before:
            #batch_size × seq_len × d_model
        #after reshape
            #batch_size × seq_len × n_heads × head_dim
        #after transpose:
            #batch_size × n_heads × seq_len × head_dim
    qh = q.reshape(batch_size, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
    kh = k.reshape(batch_size, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)
    vh = v.reshape(batch_size, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)

    #attention score
    scores = qh @ kh.transpose(0, 1, 3, 2)

    #dot products get larger when vectors are wider, divide by sqrt(head_dim) to keep attention scores from becoming too extreme
    scores = scores / math.sqrt(head_dim)

    #future-token mask
        #token 0 cannot see token 1,2,3...
        #token 1 cannot see token 2,3...
        #token 2 cannot see token 3...
    future_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)

    #put a huge negative score where attention is forbidden
    #after softmax, these become basically probability 0
    scores[:, :, future_mask] = -1e9

    #convert scores to attention probabilities
    probs = softmax(scores)

    #weighted sum of value vectors
    heads = probs @ vh

    #recombine heads back into one d_model vector per token
    combined = heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

    #final attention projection
    out = combined @ p[f"{layer}.Wo"]

    #cache values needed by backward pass
    cache = (h, qh, kh, vh, probs, combined)

    return out, cache


def attention_backward(dout, cache, p, layer, cfg):
    """
    backpropagate through attention
    returns gradient for the input hidden state
    """

    h, qh, kh, vh, probs, combined = cache

    batch_size, seq_len, d_model = h.shape

    head_dim = cfg.head_dim
    n_heads = cfg.n_heads

    grads = {}

    grads[f"{layer}.Wo"] = combined.reshape(-1, d_model).T @ dout.reshape(-1, d_model)

    #gradient flowing backward into combined
    dcombined = dout @ p[f"{layer}.Wo"].T

    #split back into heads
    dheads = dcombined.reshape(batch_size, seq_len, n_heads, head_dim).transpose(0, 2, 1, 3)

    dprobs = dheads @ vh.transpose(0, 1, 3, 2)
    dvh = probs.transpose(0, 1, 3, 2) @ dheads

    dscores = probs * (dprobs - np.sum(dprobs * probs, axis=-1, keepdims=True))

    dscores = dscores / math.sqrt(head_dim)

    dqh = dscores @ kh
    dkh = dscores.transpose(0, 1, 3, 2) @ qh

    #recombine heads
    dq = dqh.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    dk = dkh.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    dv = dvh.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)

    flat_h = h.reshape(-1, d_model)

    grads[f"{layer}.Wq"] = flat_h.T @ dq.reshape(-1, d_model)
    grads[f"{layer}.Wk"] = flat_h.T @ dk.reshape(-1, d_model)
    grads[f"{layer}.Wv"] = flat_h.T @ dv.reshape(-1, d_model)

    #gradient back into h from q/k/v paths
    dh = dq @ p[f"{layer}.Wq"].T
    dh += dk @ p[f"{layer}.Wk"].T
    dh += dv @ p[f"{layer}.Wv"].T

    return dh, grads


def forward(x, p, cfg):
    """
    run the model forward
    turns input token IDs into next-token scores
    saved intermediate values needed for backpropagation
    """

    batch_size, seq_len = x.shape

    #token embedding lookup
    #p["tok_emb"][x] replaces each token ID with its learned vector.
    token_vectors = p["tok_emb"][x]

    #position IDs
    positions = np.arange(seq_len)

    #position embedding lookup
    #the leading 1 lets NumPy broadcast it across the batch
    position_vectors = p["pos_emb"][positions][None, :, :]

    #hidden state starts as token meaning + position meaning
    h = token_vectors + position_vectors

    caches = []

    for layer in range(cfg.n_layers):
        #save h before attention for the residual connection
        h_before_attention = h

        attention_output, attention_cache = attention_forward(h, p, layer, cfg)

        #residual connection
        #this helps information and gradients flow through deep networks
        h = h_before_attention + attention_output

        #save h before MLP for the second residual connection
        h_before_mlp = h

        #MLP first projection
        pre_activation = h @ p[f"{layer}.W1"] + p[f"{layer}.b1"]

        #ReLU activation
            #negative values become 0
            #positive values stay
        mlp_hidden = np.maximum(pre_activation, 0)

        #MLP second projection
        mlp_output = mlp_hidden @ p[f"{layer}.W2"] + p[f"{layer}.b2"]

        #second residual connection
        h = h_before_mlp + mlp_output

        #save what backward pass needs
        caches.append((attention_cache, pre_activation, mlp_hidden, h_before_mlp))

    #output head
    #for every token position, produce one score for every possible next token
    logits = h @ p["Wout"] + p["bout"]

    return logits, h, caches


def backward(x, y, logits, h, caches, p, cfg):
    """
    run backpropagation
        1. calculate loss
        2. start gradient at output logits
        3. move backward through output head
        4. move backward through each transformer block
        5. collect gradients for every parameter
    """

    loss, dlogits = cross_entropy(logits, y)

    #create a gradient dictionary with the same keys/shapes as parameters
    grads = {name: np.zeros_like(value) for name, value in p.items()}

    batch_size, seq_len, d_model = h.shape

    #backward through output head
    grads["Wout"] = h.reshape(-1, d_model).T @ dlogits.reshape(-1, cfg.vocab_size)
    grads["bout"] = np.sum(dlogits, axis=(0, 1))

    #gradient flowing back into hidden state
    dh = dlogits @ p["Wout"].T

    #go backward through transformer layers
    for layer in reversed(range(cfg.n_layers)):
        attention_cache, pre_activation, mlp_hidden, h_before_mlp = caches[layer]

        #backward through MLP residual
        d_mlp_output = dh

        grads[f"{layer}.W2"] += mlp_hidden.reshape(-1, cfg.d_ff).T @ d_mlp_output.reshape(-1, cfg.d_model)
        grads[f"{layer}.b2"] += np.sum(d_mlp_output, axis=(0, 1))

        d_mlp_hidden = d_mlp_output @ p[f"{layer}.W2"].T

        #backward through ReLU
        #blocked negative values get zero gradient.
            #ReLU(x) = x if x > 0
            #ReLU(x) = 0 if x <= 0
        d_pre_activation = d_mlp_hidden * (pre_activation > 0)

        grads[f"{layer}.W1"] += h_before_mlp.reshape(-1, cfg.d_model).T @ d_pre_activation.reshape(-1, cfg.d_ff)
        grads[f"{layer}.b1"] += np.sum(d_pre_activation, axis=(0, 1))

        #add gradient through residual path
        #add gradient through MLP path
        dh = dh + d_pre_activation @ p[f"{layer}.W1"].T

        #backward through attention residual
        d_attention_output = dh

        dh_from_attention, attention_grads = attention_backward(
            dout=d_attention_output,
            cache=attention_cache,
            p=p,
            layer=layer,
            cfg=cfg,
        )

        for name, grad in attention_grads.items():
            grads[name] += grad

        #add gradient through residual path
        #add gradient through attention path
        dh = dh + dh_from_attention

    #position embedding gradients
    #the same position embedding is used for every example in the batch, so sum over the batch
    grads["pos_emb"][:seq_len] += np.sum(dh, axis=0)

    #token embedding gradients
        #token embeddings are looked up by ID
        #if the same token appears many times, all those gradients need to add into the same row of tok_emb
        #np.add.at handles repeated indexes correctly
    np.add.at(
        grads["tok_emb"],
        x.reshape(-1),
        dh.reshape(-1, cfg.d_model),
    )

    return loss, grads


def adamw_update(p, grads, state, lr, step, weight_decay=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    update parameters using AdamW
    gradients say which way to move, and this optimizer decides how to move smoothly and safely

    p: trainable parameters
    grads: gradients from backpropagation
    state: optimizer memory
    lr: learning rate
    step: current training step

    smooths the average gradient direction
    beta1 = 0.9

    smooths the average squared gradient size
    beta2 = 0.999

    tiny number to avoid division by zero
    eps = 1e-8

    gently pushes weights toward smaller values
    weight_decay = 0.01
    """

    #create optimizer memory
        #m: moving average of gradients
        #v: moving average of squared gradients
    if not state:
        state["m"] = {name: np.zeros_like(value) for name, value in p.items()}
        state["v"] = {name: np.zeros_like(value) for name, value in p.items()}

    for name in p:
        grad = grads[name]

        #update moving averages
        state["m"][name] = beta1 * state["m"][name] + (1 - beta1) * grad
        state["v"][name] = beta2 * state["v"][name] + (1 - beta2) * (grad * grad)

        #bias correction
            #early moving averages are biased toward zero
            #these corrections compensate for that
        m_hat = state["m"][name] / (1 - beta1 ** step)
        v_hat = state["v"][name] / (1 - beta2 ** step)

        #Adam update
        update = m_hat / (np.sqrt(v_hat) + eps)

        #AdamW update
        p[name] -= lr * (update + weight_decay * p[name])


def generate(prompt, tokenizer, p, cfg, max_new_tokens=100):
    """
    generate text after training
    """

    ids = tokenizer.encode(prompt)

    for _ in range(max_new_tokens):
        #only feed the latest context_length tokens
        recent_ids = ids[-cfg.context_length:]

        #Add batch dimension
        x = np.array(recent_ids, dtype=np.int64)[None, :]

        logits, _, _ = forward(x, p, cfg)

        #use only the final position to pick the next token
        next_token_logits = logits[0, -1]

        next_token_probs = softmax(next_token_logits)

        next_id = int(np.random.choice(np.arange(cfg.vocab_size), p=next_token_probs))

        ids.append(next_id)

    return tokenizer.decode(ids)


def train():
    """
    main training loop
        load config
        create tokenizer
        initialize weights
        stream batches
        forward pass
        calculate loss
        backward pass
        update weights
        repeat
    """

    cfg = Config()

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
    print(f"output_head    = {cfg.d_model} × {cfg.vocab_size}")
    print(f"head_dim       = {cfg.head_dim}")
    print(f"n_heads        = {cfg.n_heads}")
    print(f"d_ff           = {cfg.d_ff}")
    print(f"mlp_size       = {cfg.d_model} × {cfg.d_ff}")
    print(f"total_blocks   = {cfg.n_layers}")

    #initialize trainable weights
    p = init_params(cfg)

    #optimizer memory starts empty
    optimizer_state = {}

    #streaming batch generator
    #it yields:
        #x = input token IDs
        #y = next-token target IDs
    batches = batch_stream(
        textsource=cfg.textsource,
        tokenizer=tokenizer,
        context_length=cfg.context_length,
        batch_size=cfg.batch_size,
    )

    for step in range(1, cfg.train_steps + 1):
        x, y = next(batches)

        #forward: make predictions
        logits, h, caches = forward(x, p, cfg)

        #backward: calculate loss and gradients.
        loss, grads = backward(x, y, logits, h, caches, p, cfg)

        #optimizer: update weights.
        adamw_update(
            p=p,
            grads=grads,
            state=optimizer_state,
            lr=cfg.learning_rate,
            step=step,
        )

        if step % 100 == 0:
            print(f"step={step} loss={loss:.4f}")

    print(generate("", tokenizer, p, cfg))


if __name__ == "__main__":
    train()
