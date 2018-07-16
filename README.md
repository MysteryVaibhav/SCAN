# SCAN : Stacked Cross Attention for Image-Text Matching
This is an implementation for the paper: https://arxiv.org/abs/1803.08024.

_master branch_: t2i model as described in the paper.
_i2t branch_: i2t model as described in the paper.

To train the model
```
python main.py --mode 0
```

To load and test the best model
```
python main.py --mode 1
```

Note: Look up main.py for passing additional arguments while training.

Results on _fast branch_ which is a faster version of master and is a t2i model:

|r@1 (t2i)   |r@5 (t2i)   |r@10 (t2i)  |r@1 (i2t)  |r@5 (i2t)  |r@10 (i2t) |
|------|------|------|-----|-----|-----|
|0.48314|0.7674|0.824|0.624|0.878|0.929|


