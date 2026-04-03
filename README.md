# nanoGPT Kotlin

`nanoGPT-kotlin` is a Kotlin/JVM port of the core [`karpathy/nanoGPT`](https://github.com/karpathy/nanoGPT) workflow. It prepares character-level datasets, trains decoder-only GPT models, resumes training with optimizer state, inspects checkpoints, samples text, and imports pretrained GPT-2 weights into native Kotlin checkpoints.

The runtime is [DJL](https://djl.ai/) with the PyTorch engine, so the product stays on the JVM while using a mature tensor backend.

## What This Product Includes

- A single CLI entrypoint with `prepare`, `train`, `sample`, `inspect`, and `import-gpt2`
- Scratch training and pretrained GPT-2 initialization via `--init_from`
- Causal self-attention, GELU MLP blocks, dropout, pre-layer normalization, and weight tying
- Character-level dataset preparation to `train.bin`, `val.bin`, and `vocab.txt`
- Automatic checkpoint layout with `best/` and `latest/`
- Resume support with model weights, optimizer moments, update counts, and RNG state
- A shaded runnable jar from `mvn package`

## Parity Notes

This project now covers the core single-process `nanoGPT` workflow on the JVM:

- scratch training
- GPT-2 initialization
- checkpoint save/load
- optimizer-state resume
- text generation

Important behavior note: resumed runs re-evaluate the current iteration after restart, matching upstream `nanoGPT`. Because of that, resumed `train loss` and `val loss` logs around the restart boundary can differ from a never-stopped run even when subsequent training updates are identical.

## Current Limits

- Character-level data preparation is included; GPT-2-tokenized dataset preparation is not
- DDP, `torch.compile`, and W&B logging are still out of scope
- GPT-2 import expects Hugging Face safetensors for `gpt2`, `gpt2-medium`, `gpt2-large`, or `gpt2-xl`

## Build

```bash
cd nanoGPT-kotlin
mvn package
```

This produces:

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
- `import-gpt2`

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
  --init_from=scratch \
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

Resume from a previous run by pointing `resume_from` at either the root output directory or a specific checkpoint slot:

```bash
java -jar target/nanogpt-kotlin-1.0.0.jar train \
  --dataset_dir=data/shakespeare_char \
  --out_dir=out-shakespeare-char-resumed \
  --init_from=resume \
  --resume_from=out-shakespeare-char \
  --device=cpu \
  --max_iters=8000
```

If you pass the root output directory, resume automatically prefers `latest/`.

Resume restores:

- model weights
- optimizer mean / variance tensors
- optimizer update counts
- training RNG state
- evaluation RNG state

## Initialize From GPT-2

You can initialize training directly from a pretrained GPT-2 checkpoint:

```bash
java -jar target/nanogpt-kotlin-1.0.0.jar train \
  --dataset_dir=data/shakespeare_char \
  --out_dir=out-gpt2-init \
  --init_from=gpt2 \
  --block_size=256 \
  --dropout=0.0 \
  --max_iters=1000
```

Supported model types:

- `gpt2`
- `gpt2-medium`
- `gpt2-large`
- `gpt2-xl`

## Import GPT-2 Into a Native Kotlin Checkpoint

To create a reusable native checkpoint from Hugging Face weights:

```bash
java -jar target/nanogpt-kotlin-1.0.0.jar import-gpt2 \
  --model_type=gpt2 \
  --output_dir=checkpoints/gpt2
```

If you already downloaded the safetensors files locally:

```bash
java -jar target/nanogpt-kotlin-1.0.0.jar import-gpt2 \
  --model_type=gpt2 \
  --source_dir=/path/to/huggingface-model-dir \
  --output_dir=checkpoints/gpt2
```

This writes a standard checkpoint directory that can be used by `sample`, `inspect`, or `train --init_from=resume`.

## Inspect a Checkpoint

```bash
java -jar target/nanogpt-kotlin-1.0.0.jar inspect \
  --checkpoint_dir=out-shakespeare-char
```

If you pass the root output directory, inspection automatically resolves to `best/`.

Inspection includes model metadata and whether optimizer state is present.

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
