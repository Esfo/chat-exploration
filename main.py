from word_survival import word_survival
from training import Config, train


#=== token survival config ===

#how many consecutive rounds a leaf may sit at score 0 (absent) before it dies. a leaf
#gains +1 when it appears and loses 1 when absent; once its score hits 0, this is the
#grace period of further absences it gets before being removed.
survivalrounds = 1

textsource = '/home/sfo/store/gutenberg/gutenbooks/'

tokenoutput = '/home/sfo/data/models/tokens/text-chunks.jsonl'
tokenoutput = False

tokeninput = '/home/sfo/data/models/tokens/text-chunks.jsonl'

coveragetest = True
#coveragetest = False


#=== training config ===

training = True
#training = False

#number of tokens per training chunk
context_length = 128

#width of the model's internal token vector
d_model = 256

#number of transformer blocks
n_layers = 2

#width of one attention head (n_heads = d_model / head_dim)
head_dim = 64

#controls how wide the MLP part gets inside each transformer block
mlp_multiplier = 4.0

#save the encoded corpus (the whole corpus as one integer token-ID tensor, NOT
#the vocabulary jsonl) to this .pt file so only the first run pays the tokenising
#cost; later runs load it instantly. set to False to re-tokenise every run.
corpus_cache = '/home/sfo/data/models/tokens/corpus.pt'
#corpus_cache = False

#use rotary position encoding (RoPE) instead of a learned position-embedding
#table. RoPE bakes position into attention by rotating queries/keys, generalises
#better, and needs no position table. requires head_dim to be even.
use_rope = True

#use a SwiGLU gated MLP instead of the plain GELU MLP. modern standard, small
#quality bump, slightly more params/compute per block. set False for plain GELU.
use_swiglu = True

#number of chunks trained together in one update
batch_size = 8

#how large each training update is (the peak LR the schedule warms up to).
#adamw likes a much smaller LR than plain gradient descent did; ~3e-4 is a
#sane default. if you switch optimizer to 'sgd', raise this a lot (e.g. 0.5).
learning_rate = 3e-4

#how often (in steps) to print the training loss to stdout
log_every = 10

#where to save the trained model.
#saved on the periodic checkpoint, when the loss target is reached, and on
#interruption (ctrl-c). set to False to skip saving entirely.
model_output = '/home/sfo/data/models/model.pt'
#model_output = False

#path to an existing saved model to continue training from.
#set to a saved .pt path to resume; False to start from fresh random weights.
resume_from = False
#resume_from = '/home/sfo/data/models/model.pt'

#early-stopping target: training runs (no fixed step count) until the average
#loss over the last target_window steps drops to/below target_loss.
#set target_loss to False to train forever until interrupted.
target_loss = 0.1
target_window = 100

#save a checkpoint every this many steps so progress survives a crash.
checkpoint_every = 200


#=== hardware / speed ===

#where the model trains: 'auto' uses the GPU if one is visible, else the CPU.
#force 'cuda' or 'cpu' to override.
device = 'auto'

#background worker processes that prepare batches in parallel while the GPU
#trains. 0 loads in the main process. a handful of your CPU cores is a good
#start (you have plenty). lower it if you hit memory pressure.
num_workers = 8

#mixed-precision training: do the heavy matmuls in fp16 on the GPU for ~2x
#speed and less VRAM. ignored on CPU. set False to train in full fp32.
use_amp = True


#=== optimizer / schedule ===

#'adamw' (recommended modern default) or 'sgd' (plain gradient descent, like
#the original setup but far more sensitive to learning_rate).
optimizer = 'adamw'

#adamw weight decay (mild regularisation). ignored for sgd.
weight_decay = 0.1

#clip the global gradient norm to this before each step so one bad batch can't
#blow the weights up. set to False to disable.
grad_clip = 1.0

#linear LR warmup: ramp 0 -> learning_rate over this many steps, then decay.
warmup_steps = 100

#cosine decay horizon: LR eases learning_rate -> min_lr over this many steps,
#then holds at min_lr. set to False for a constant LR.
lr_decay_steps = 5000
min_lr = 1e-4


#=== validation ===

#fraction of the corpus held out (never trained on) to measure validation
#loss. set to 0 to disable validation entirely.
val_fraction = 0.05

#run a validation pass every this many steps, averaging over val_batches.
val_every = 200
val_batches = 20


#=== pipeline ===

#the actual work lives under the __main__ guard below. this matters because the
#DataLoader's worker processes import this module to get at the code; without
#the guard, every worker would re-run the whole pipeline (re-tokenising the
#entire corpus once per worker). with it, workers import the config only and
#receive the already-encoded data from the parent.

def run():
    if tokenoutput:
        tokensfile = word_survival(textsource, tokenoutput, survivalrounds, coveragetest)
        print('tokens written to', tokensfile)
    else:
        tokensfile = tokeninput
        print('using existing tokens at', tokensfile)

    if not training:
        print('training disabled')
        return

    cfg = Config(
        context_length=context_length,
        d_model=d_model,
        n_layers=n_layers,
        head_dim=head_dim,
        mlp_multiplier=mlp_multiplier,
        use_rope=use_rope,
        use_swiglu=use_swiglu,
        #paths come from this gateway, not from training.py
        tokenpath=str(tokensfile),
        textsource=textsource,
        corpus_cache=str(corpus_cache) if corpus_cache else "",
        batch_size=batch_size,
        learning_rate=learning_rate,
        log_every=log_every,
        #saving / resuming / early-stop are all optional; False disables each.
        model_output=str(model_output) if model_output else "",
        resume_from=str(resume_from) if resume_from else "",
        target_loss=float(target_loss) if target_loss else 0.0,
        target_window=target_window,
        checkpoint_every=checkpoint_every,
        #hardware / speed
        device=device,
        num_workers=num_workers,
        use_amp=use_amp,
        #optimizer / schedule
        optimizer=optimizer,
        weight_decay=weight_decay,
        grad_clip=float(grad_clip) if grad_clip else 0.0,
        warmup_steps=warmup_steps,
        lr_decay_steps=int(lr_decay_steps) if lr_decay_steps else 0,
        min_lr=min_lr,
        #validation
        val_fraction=val_fraction,
        val_every=val_every,
        val_batches=val_batches,
    )
    train(cfg)


if __name__ == "__main__":
    run()
