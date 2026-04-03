# nanoGPT Kotlin

This project is a Kotlin/JVM port of the core ideas in `karpathy/nanoGPT`. It includes a GPT model comparable to `model.py`, a training CLI comparable to `train.py`, a sampling CLI comparable to `sample.py`, and a simple character-level dataset preparation CLI.

The implementation runs on the JVM using the [DJL](https://djl.ai/) PyTorch engine. It is not a literal 1:1 translation of the original Python code. Instead, it rebuilds the same model structure and workflow in Kotlin.

## Features

- Train a GPT model from scratch
- Causal self-attention, pre-LN, GELU, residual connections, and dropout
- Weight tying between token embeddings and the language-model head
- Read `train.bin` and `val.bin` datasets stored as raw `uint16`
- Text generation with temperature and top-k sampling
- Prepare character-level datasets from plain text

## Not Implemented Yet

- DDP or multi-node training
- `torch.compile`
- GPT-2 pretrained weight import
- Exact resume support including optimizer state
- W&B logging

## Build

```bash
cd nanoGPT-kotlin
mvn compile
```

On first run, DJL may download the required PyTorch runtime for your platform. If you need a fully offline setup, add the appropriate runtime dependency to `pom.xml`.

## Prepare a Character Dataset

You can generate `train.bin`, `val.bin`, and `vocab.txt` from any plain text file.

```bash
mvn -q -Dexec.mainClass=dev.naoki.nanogpt.PrepareTextCli exec:java \
  -Dexec.args="--input=/path/to/input.txt --output_dir=data/shakespeare_char"
```

## Train

```bash
mvn -q -Dexec.mainClass=dev.naoki.nanogpt.TrainCli exec:java \
  -Dexec.args="configs/train-shakespeare-char.properties"
```

The CLI supports both `.properties` config files and `--key=value` overrides.

Example:

```bash
mvn -q -Dexec.mainClass=dev.naoki.nanogpt.TrainCli exec:java \
  -Dexec.args="configs/train-shakespeare-char.properties --device=cpu --max_iters=2000"
```

## Sample

```bash
mvn -q -Dexec.mainClass=dev.naoki.nanogpt.SampleCli exec:java \
  -Dexec.args="--checkpoint_dir=out-shakespeare-char --start=$'\n' --num_samples=3 --max_new_tokens=300"
```

If `vocab.txt` is present in the checkpoint directory, the sampler uses the character-level codec. Otherwise, it falls back to a GPT-style tokenizer.

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
