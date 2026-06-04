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

#number of chunks trained together in one update
batch_size = 8

#how large each training update is
learning_rate = 0.5

#number of training steps to run
train_steps = 2000

#how often (in steps) to print the training loss to stdout
log_every = 1

#where to save the trained model when training finishes.
#set to a path like '/home/sfo/data/models/model.npz' to save; False to skip.
model_output = '/home/sfo/data/models/model.npz'
#model_output = False

#path to an existing saved model to continue training from.
#set to a saved .npz path to resume; False to start from fresh random weights.
resume_from = False
#resume_from = '/home/sfo/data/models/model.npz'

#optional early-stopping target loss. training stops once loss drops to/below
#this value. set to False to always run the full train_steps.
target_loss = False
#target_loss = 0.5


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
        #saving / resuming / early-stop are all optional; False disables each.
        model_output=str(model_output) if model_output else "",
        resume_from=str(resume_from) if resume_from else "",
        target_loss=float(target_loss) if target_loss else 0.0,
    )
    train(cfg)
else:
    print('training disabled')
