# APATH-Ano

### Data prepare
Download the HPH dataset at 

### Training process

First, use the on-the-shelf VLM for zero-shot inference on the training set.
You can adjust the keeping ratio M.
```
python zero_shot_HPH_un.py --gpu 0
```
