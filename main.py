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
context_length = 256

#width of the model's internal token vector
#means every token becomes a vector with this many numbers
d_model = 384

#number of transformer blocks
n_layers = 4

#width of one attention head (n_heads = d_model / head_dim)
head_dim = 64

#controls how wide the MLP part gets inside each transformer block
mlp_multiplier = 2.67

#number of chunks trained together in one update
batch_size = 16

#AdamW base learning rate. AdamW adapts per-parameter; the scheduler controls
#the global learning rate over time.
learning_rate = 3e-4
min_learning_rate = 3e-5

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

#early-stopping target: training stops once the average loss over the last
#target_window steps drops to/below target_loss. set target_loss to False to
#train until max_steps or interruption.
target_loss = False
#target_loss = 0.1
target_window = 100

#maximum number of optimizer updates to run.
max_steps = 5000

#save a checkpoint every this many steps so progress survives a crash.
checkpoint_every = 200


#=== modern training controls ===

device = 'auto'          # auto, cuda, cpu
dtype = 'auto'           # auto, float32, float16, bfloat16
compile_model = True     # torch.compile when available

optimizer = 'adamw'
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95

grad_clip = 1.0

#cosine = warmup + cosine decay. plateau = reduce LR when validation stops
#improving. constant = no schedule.
scheduler = 'cosine'
#scheduler = 'plateau'
warmup_steps = 100
plateau_factor = 0.5
plateau_patience = 3

#data loading / validation
num_workers = 4
prefetch_factor = 2
pin_memory = True
validation_fraction = 0.05
eval_every = 200
eval_batches = 20
seed = 1

#architecture switches
norm_type = 'rmsnorm'    # rmsnorm or layernorm
use_rope = True
use_swiglu = True
tie_weights = True
dropout = 0.0

#sampling controls used for the sample printed after training and by chat.py
sample_max_new_tokens = 100
sample_temperature = 0.8
sample_top_k = 50
sample_top_p = 0.95
sample_greedy = False


#=== pipeline ===

if tokenoutput:
    tokensfile = word_survival(textsource, tokenoutput, survivalrounds, coveragetest)
    print('tokens written to', tokensfile)
else:
    tokensfile = tokeninput
    print('using existing tokens at', tokensfile)

if training:
    cfg = Config(
        context_length=context_length,
        d_model=d_model,
        n_layers=n_layers,
        head_dim=head_dim,
        mlp_multiplier=mlp_multiplier,
        #paths come from this gateway, not from training.py
        tokenpath=str(tokensfile),
        textsource=textsource,
        batch_size=batch_size,
        learning_rate=learning_rate,
        log_every=log_every,
        #saving / resuming / early-stop are all optional; False disables each.
        model_output=str(model_output) if model_output else "",
        resume_from=str(resume_from) if resume_from else "",
        target_loss=float(target_loss) if target_loss else 0.0,
        target_window=target_window,
        checkpoint_every=checkpoint_every,
        max_steps=max_steps,
        device=device,
        dtype=dtype,
        compile_model=compile_model,
        optimizer=optimizer,
        weight_decay=weight_decay,
        beta1=beta1,
        beta2=beta2,
        grad_clip=grad_clip,
        scheduler=scheduler,
        warmup_steps=warmup_steps,
        min_learning_rate=min_learning_rate,
        plateau_factor=plateau_factor,
        plateau_patience=plateau_patience,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        validation_fraction=validation_fraction,
        eval_every=eval_every,
        eval_batches=eval_batches,
        seed=seed,
        norm_type=norm_type,
        use_rope=use_rope,
        use_swiglu=use_swiglu,
        tie_weights=tie_weights,
        dropout=dropout,
        sample_max_new_tokens=sample_max_new_tokens,
        sample_temperature=sample_temperature,
        sample_top_k=sample_top_k,
        sample_top_p=sample_top_p,
        sample_greedy=sample_greedy,
    )
    train(cfg)
else:
    print('training disabled')
