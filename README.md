# SCAN : Stacked Cross Attention for Image-Text Matching
This is an implementation for the paper: https://arxiv.org/abs/1803.08024

To train the model
```
python main.py --mode 0
```

To load and test the best model
```
python main.py --mode 1
```

Note: Look up main.py for passing additional arguments while training.

Intermediate results for text-to-image:

|r@1   |r@5   |r@10  |
|------|------|------|
|0.4536|0.7354|0.7976|
