# APATH-Ano

### Data prepare
Download the HPH dataset at 

### Training process

First, use the on-the-shelf VLM for zero-shot inference with our proposed ODP to filter out noisy samples on the training set.
```
python zero_shot_HPH_ODP.py --gpu 0
```
Second, after obtaining high-quality pseudo-labels, you can train a classification network via LoRA.
```
python train_pseudo.py --gpu 0
```
