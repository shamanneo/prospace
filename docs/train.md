# Training

```bash
python scripts/prepare_train.py --set 100k --polling      # once, after the data of data.md
bash scripts/train.sh prospace-vlm 0,1,2,3                # prints the command to evaluate the result
```

| Model | Trainer | Settings |
|---|---|---|
| `prospace-vlm` | native streaming, with geometry | LoRA r 16, lr 1e-4, 16 streams per step, one epoch |
| `qwen2.5-vl-7b-streaming` | native streaming, without geometry | same |
| `qwen3-vl-8b-polling` | prefix polling | LoRA r 16, lr 1e-4, 64 samples per step, one epoch |
| `qwen2.5-vl-7b-polling` | prefix polling | same |

- The third argument chooses the set: `100k` (default) or `400k` (the extended set).
- Any number of GPUs works: the steps are accumulated to the same batch size.
- The best checkpoint (lowest validation loss) is `runs/<model>/best` (`adapter_best` for prefix polling). Run the same command again to resume.
- `WANDB=1` logs the curves to wandb (`pip install wandb`, entity from `WANDB_ENTITY`).

**Native streaming**: a decision token after every frame, `wait` before the window and `respond` inside it; absence streams are `wait` to the end.

**Prefix polling**: samples of the form (frames 0..t, question) → Yes / No, built from the same streams by `prepare_train.py --polling`.
