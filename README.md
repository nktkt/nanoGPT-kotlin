# nanoGPT Kotlin

`nanoGPT-kotlin` is a Kotlin/JVM implementation of the core nanoGPT workflow. It can prepare character-level datasets, train a causal decoder-only transformer, resume from saved weights, inspect checkpoints, and generate text from saved models.

The runtime uses [DJL](https://djl.ai/) with the PyTorch engine, so the project stays on the JVM while still using a mature tensor backend.

## What This Product Includes

- A runnable CLI with `prepare`, `train`, `sample`, and `inspect` commands
- Scratch training for GPT-style decoder models
- Character-level dataset preparation to `train.bin`, `val.bin`, and `vocab.txt`
- Weight tying, causal self-attention, GELU MLP blocks, dropout, and pre-layer normalization
- Automatic checkpoint layout with `best/` and `latest/`
- Weight-based resume support from `latest/` checkpoints
- Executable shaded jar output from `mvn package`

## Current Limits

- Resume restores model weights and training metadata, but not optimizer state
- Character-level preparation is included; GPT-2-tokenized dataset preparation is not
- DDP, `torch.compile`, pretrained GPT-2 import, and W&B logging are still out of scope

## Build

```bash
cd nanoGPT-kotlin
mvn package
```

This produces a runnable jar at:

```bash
target/nanogpt-kotlin-1.0.0.jar
```

You can also run the CLI without packaging:

```bash
mvn -q -Dexec.mainClass=dev.naoki.nanogpt.NanoGptCli exec:java -Dexec.args="--help"
```

On first run, DJL may download the required PyTorch runtime for your platform.

## CLI Overview

```bash
java -jar target/nanogpt-kotlin-1.0.0.jar --help
```

Available commands:

- `prepare`
- `train`
- `sample`
- `inspect`

## Prepare a Dataset

```bash
java -jar target/nanogpt-kotlin-1.0.0.jar prepare \
  --input=/path/to/input.txt \
  --output_dir=data/shakespeare_char \
  --train_split=0.9
```

Output files:

- `train.bin`
- `val.bin`
- `vocab.txt`

## Train

You can train from a config file:

```bash
java -jar target/nanogpt-kotlin-1.0.0.jar train \
  configs/train-shakespeare-char.properties
```

Or with explicit overrides:

```bash
java -jar target/nanogpt-kotlin-1.0.0.jar train \
  --dataset_dir=data/shakespeare_char \
  --out_dir=out-shakespeare-char \
  --device=cpu \
  --batch_size=64 \
  --block_size=256 \
  --n_layer=6 \
  --n_head=6 \
  --n_embd=384 \
  --max_iters=5000
```

Training writes checkpoints to:

- `out-shakespeare-char/latest`
- `out-shakespeare-char/best`

## Resume Training

Resume from a previous run by pointing `resume_from` at the output directory or directly at a checkpoint slot:

```bash
java -jar target/nanogpt-kotlin-1.0.0.jar train \
  --dataset_dir=data/shakespeare_char \
  --out_dir=out-shakespeare-char-resumed \
  --resume_from=out-shakespeare-char \
  --device=cpu \
  --max_iters=8000
```

If you pass the root output directory, resume automatically prefers `latest/`.

## Inspect a Checkpoint

```bash
java -jar target/nanogpt-kotlin-1.0.0.jar inspect \
  --checkpoint_dir=out-shakespeare-char
```

If you pass the root output directory, inspection automatically resolves to `best/`.

## Sample

Generate text from a checkpoint:

```bash
java -jar target/nanogpt-kotlin-1.0.0.jar sample \
  --checkpoint_dir=out-shakespeare-char \
  --start=$'\n' \
  --num_samples=3 \
  --max_new_tokens=300 \
  --temperature=0.8 \
  --top_k=200
```

You can also load the prompt from a file:

```bash
java -jar target/nanogpt-kotlin-1.0.0.jar sample \
  --checkpoint_dir=out-shakespeare-char \
  --start_file=/path/to/prompt.txt
```

If you pass the root output directory, sampling automatically resolves to `best/`.

## Example Config

The repository includes a starter config:

```bash
configs/train-shakespeare-char.properties
```

## Push to a New GitHub Repository

If you want to publish this project to a brand-new GitHub repository:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

If the repository already exists on GitHub and you prefer HTTPS:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```
