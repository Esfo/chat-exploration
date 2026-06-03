"""
STAGE 11 — THE TRAINING LOOP (everything together)
==================================================

Source: `train()` in training.py

WHAT THIS STAGE DOES
--------------------
This is the conductor. It wires every previous stage into one repeating cycle
and runs it `train_steps` times:

    for each step:
        x, y          = next batch          (inputs and the next-token targets)
        logits, cache = forward(x)          stages 2-6  (embed -> attn -> mlp -> head)
        loss, grads   = backward(...)       stages 7-9  (cross-entropy -> backprop)
        adamw_update(p, grads)              stage 10    (take a learning step)

Each loop nudges the weights so the next forward pass predicts the data a little
better. Loss should trend DOWN.

WHY THIS PROVES THE WHOLE THING WORKS
-------------------------------------
Below we build the same tiny transformer as the backprop sim and train it on a
trivial, learnable pattern (predict the next number in a repeating sequence). If
all the math from stages 1-10 is correct, the loss will fall and the model will
start predicting the pattern. We watch the loss curve drop — exactly the
`step= loss=` line training.py prints to stdout.

WHAT'S DELIBERATELY LEFT OUT
----------------------------
Real tokenization/text streaming (the user asked to skip those). We feed a
synthetic repeating sequence so the learning dynamics are visible in seconds.
"""

import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

rng = np.random.default_rng(0)
vocab, d_model, d_ff, seq, batch = 6, 16, 64, 8, 16
n_heads, head_dim, n_layers = 2, 8, 1

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x); return e / np.sum(e, axis=-1, keepdims=True)

def init():
    p = {"tok_emb": rng.normal(0,0.02,(vocab,d_model)),
         "pos_emb": rng.normal(0,0.02,(seq,d_model))}
    for L in range(n_layers):
        for nm in ("Wq","Wk","Wv","Wo"):
            p[f"{L}.{nm}"] = rng.normal(0,1/math.sqrt(d_model),(d_model,d_model))
        p[f"{L}.W1"]=rng.normal(0,1/math.sqrt(d_model),(d_model,d_ff)); p[f"{L}.b1"]=np.zeros(d_ff)
        p[f"{L}.W2"]=rng.normal(0,1/math.sqrt(d_ff),(d_ff,d_model));     p[f"{L}.b2"]=np.zeros(d_model)
    p["Wout"]=rng.normal(0,1/math.sqrt(d_model),(d_model,vocab)); p["bout"]=np.zeros(vocab)
    return p

def forward(x, p):
    B,S = x.shape
    h = p["tok_emb"][x] + p["pos_emb"][np.arange(S)][None]
    caches=[]
    for L in range(n_layers):
        hba=h
        q=h@p[f"{L}.Wq"]; k=h@p[f"{L}.Wk"]; v=h@p[f"{L}.Wv"]
        qh=q.reshape(B,S,n_heads,head_dim).transpose(0,2,1,3)
        kh=k.reshape(B,S,n_heads,head_dim).transpose(0,2,1,3)
        vh=v.reshape(B,S,n_heads,head_dim).transpose(0,2,1,3)
        sc=(qh@kh.transpose(0,1,3,2))/math.sqrt(head_dim)
        mask=np.triu(np.ones((S,S),bool),1); sc[:,:,mask]=-1e9
        pr=softmax(sc); hd=pr@vh
        comb=hd.transpose(0,2,1,3).reshape(B,S,d_model)
        h=hba+comb@p[f"{L}.Wo"]
        hbm=h
        pre=h@p[f"{L}.W1"]+p[f"{L}.b1"]; hid=np.maximum(pre,0)
        h=hbm+hid@p[f"{L}.W2"]+p[f"{L}.b2"]
        caches.append(((hba,qh,kh,vh,pr,comb),pre,hid,hbm))
    logits=h@p["Wout"]+p["bout"]
    return logits,h,caches

def backward(x,y,logits,h,caches,p):
    pr=softmax(logits); B,S,V=logits.shape
    fp=pr.reshape(B*S,V); ft=y.reshape(B*S)
    loss=-np.mean(np.log(fp[np.arange(B*S),ft]+1e-12))
    d=fp.copy(); d[np.arange(B*S),ft]-=1; d/=B*S; dlogits=d.reshape(B,S,V)
    g={n:np.zeros_like(v) for n,v in p.items()}
    g["Wout"]=h.reshape(-1,d_model).T@dlogits.reshape(-1,V); g["bout"]=dlogits.sum((0,1))
    dh=dlogits@p["Wout"].T
    for L in reversed(range(n_layers)):
        (acache,pre,hid,hbm)=caches[L]
        g[f"{L}.W2"]+=hid.reshape(-1,d_ff).T@dh.reshape(-1,d_model); g[f"{L}.b2"]+=dh.sum((0,1))
        dhid=dh@p[f"{L}.W2"].T; dpre=dhid*(pre>0)
        g[f"{L}.W1"]+=hbm.reshape(-1,d_model).T@dpre.reshape(-1,d_ff); g[f"{L}.b1"]+=dpre.sum((0,1))
        dh=dh+dpre@p[f"{L}.W1"].T
        (hh,qh,kh,vh,prb,comb)=acache
        g[f"{L}.Wo"]+=comb.reshape(-1,d_model).T@dh.reshape(-1,d_model)
        dcomb=dh@p[f"{L}.Wo"].T
        dheads=dcomb.reshape(B,S,n_heads,head_dim).transpose(0,2,1,3)
        dprobs=dheads@vh.transpose(0,1,3,2); dvh=prb.transpose(0,1,3,2)@dheads
        dsc=prb*(dprobs-np.sum(dprobs*prb,axis=-1,keepdims=True)); dsc/=math.sqrt(head_dim)
        dqh=dsc@kh; dkh=dsc.transpose(0,1,3,2)@qh
        dq=dqh.transpose(0,2,1,3).reshape(B,S,d_model)
        dk=dkh.transpose(0,2,1,3).reshape(B,S,d_model)
        dv=dvh.transpose(0,2,1,3).reshape(B,S,d_model)
        fh=hh.reshape(-1,d_model)
        g[f"{L}.Wq"]+=fh.T@dq.reshape(-1,d_model); g[f"{L}.Wk"]+=fh.T@dk.reshape(-1,d_model); g[f"{L}.Wv"]+=fh.T@dv.reshape(-1,d_model)
        dh=dh+dq@p[f"{L}.Wq"].T+dk@p[f"{L}.Wk"].T+dv@p[f"{L}.Wv"].T
    g["pos_emb"][:S]+=dh.sum(0)
    np.add.at(g["tok_emb"], x.reshape(-1), dh.reshape(-1,d_model))
    return loss,g

def adamw(p,g,st,lr,t,wd=0.01,b1=0.9,b2=0.999,eps=1e-8):
    if not st:
        st["m"]={n:np.zeros_like(v) for n,v in p.items()}
        st["v"]={n:np.zeros_like(v) for n,v in p.items()}
    for n in p:
        st["m"][n]=b1*st["m"][n]+(1-b1)*g[n]
        st["v"][n]=b2*st["v"][n]+(1-b2)*(g[n]*g[n])
        mh=st["m"][n]/(1-b1**t); vh=st["v"][n]/(1-b2**t)
        p[n]-=lr*(mh/(np.sqrt(vh)+eps)+wd*p[n])

# learnable pattern: repeating 0,1,2,3,4,5,0,1,... model must learn "next = (cur+1)%vocab"
def make_batch():
    starts=rng.integers(0,vocab,batch)
    base=(np.arange(seq+1)[None,:]+starts[:,None])%vocab
    return base[:,:-1], base[:,1:]

print("="*70); print("TRAINING LOOP SIMULATION (capstone)"); print("="*70)
print("\nTask: learn the repeating sequence 0->1->2->3->4->5->0 ...")
p=init(); st={}; losses=[]
for step in range(1,401):
    x,y=make_batch()
    logits,h,caches=forward(x,p)
    loss,g=backward(x,y,logits,h,caches,p)
    adamw(p,g,st,lr=3e-3,t=step)
    losses.append(loss)
    if step==1 or step%50==0:
        print(f"  step={step:4d} loss={loss:.4f}")

# test the trained model
xt=np.array([[0,1,2,3,4,5,0,1]]); lg,_,_=forward(xt,p)
pred=lg[0,-1].argmax()
print(f"\nafter training, given ...0,1 the model predicts next = {pred} (correct = 2)")
print(f"loss fell from {losses[0]:.3f} to {losses[-1]:.3f}  "
      f"(random-guess loss would be ln(vocab)={math.log(vocab):.3f})")

# --- VISUALIZATION ---
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(losses, color="#4C72B0")
axes[0].axhline(math.log(vocab), color="#C44E52", ls="--", label=f"random guess = ln({vocab})")
axes[0].set_title("Training loss curve\n(forward+backward+update, repeated)")
axes[0].set_xlabel("step"); axes[0].set_ylabel("cross-entropy loss"); axes[0].legend()

probs=softmax(lg[0])
im=axes[1].imshow(probs, aspect="auto", cmap="magma", vmin=0, vmax=1)
axes[1].set_title("Trained predictions for input 0,1,2,3,4,5,0,1\n(bright = confident next token)")
axes[1].set_xlabel("predicted next token id"); axes[1].set_ylabel("position in sequence")
axes[1].set_xticks(range(vocab))
plt.colorbar(im, ax=axes[1], fraction=0.046)

plt.tight_layout()
out=os.path.join(os.path.dirname(__file__),"visualizations","11_training_loop.png")
plt.savefig(out,dpi=110)
print(f"\nVisualization saved -> {out}")
