from word_survival import word_survival
from training import Config, train


#=== word survival config ===

survivalrounds = 50

textsource = '/home/sfo/store/gutenberg/gutenbooks/'

tokenoutput = '/home/sfo/data/models/tokens/text-chunks.jsonl'
#tokenoutput = False

#tokeninput = '/home/sfo/data/models/tokens/text-chunks.jsonl'

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

#number of chunks trained together in one update
batch_size = 8

#how large each training update is
learning_rate = 1e-3

#number of training steps to run
train_steps = 2000

#how often (in steps) to print the training loss to stdout
log_every = 100


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
        train_steps=train_steps,
        log_every=log_every,
    )
    train(cfg)
else:
    print('training disabled')
