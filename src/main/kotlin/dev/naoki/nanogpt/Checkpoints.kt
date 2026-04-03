package dev.naoki.nanogpt

import ai.djl.Model
import java.nio.file.Files
import java.nio.file.Path
import java.util.Properties

data class CheckpointMetadata(
    val directory: Path,
    val modelConfig: GptConfig,
    val trainingState: TrainingState?,
    val codec: TextCodec,
)

object Checkpoints {
    fun bestDir(outDir: Path): Path = outDir.resolve(CheckpointFiles.BEST_DIR)

    fun latestDir(outDir: Path): Path = outDir.resolve(CheckpointFiles.LATEST_DIR)

    fun resolveForSampling(path: Path): Path {
        return resolveExisting(path, listOf(path, bestDir(path), latestDir(path)))
    }

    fun resolveForResume(path: Path): Path {
        return resolveExisting(path, listOf(path, latestDir(path), bestDir(path)))
    }

    fun loadMetadata(path: Path): CheckpointMetadata {
        val resolved = resolveForSampling(path)
        return CheckpointMetadata(
            directory = resolved,
            modelConfig = loadModelConfig(resolved),
            trainingState = loadTrainingState(resolved),
            codec = CharacterCodec.maybeLoad(resolved) ?: Gpt2Codec(),
        )
    }

    fun loadModelConfig(path: Path): GptConfig {
        val resolved = resolveForSampling(path)
        val props = PropertiesIO.load(resolved.resolve(CheckpointFiles.MODEL_PROPERTIES))
        return GptConfig(
            blockSize = props.getProperty("block_size").toInt(),
            vocabSize = props.getProperty("vocab_size").toInt(),
            nLayer = props.getProperty("n_layer").toInt(),
            nHead = props.getProperty("n_head").toInt(),
            nEmbd = props.getProperty("n_embd").toInt(),
            dropout = props.getProperty("dropout").toFloat(),
            bias = props.getProperty("bias").toBoolean(),
        )
    }

    fun loadTrainingState(path: Path): TrainingState? {
        val resolved = resolveForSampling(path)
        val statePath = resolved.resolve(CheckpointFiles.STATE_PROPERTIES)
        if (!Files.exists(statePath)) {
            return null
        }
        val props = PropertiesIO.load(statePath)
        return TrainingState(
            iter = props.getProperty("iter").toInt(),
            bestValLoss = props.getProperty("best_val_loss").toFloat(),
            trainRngState = props.getProperty("train_rng_state", "0").toLong(),
            trainEvalRngState = props.getProperty("train_eval_rng_state", "0").toLong(),
            valEvalRngState = props.getProperty("val_eval_rng_state", "0").toLong(),
        )
    }

    fun save(
        model: Model,
        outputDir: Path,
        modelConfig: GptConfig,
        codec: TextCodec,
        trainingState: TrainingState,
    ) {
        Files.createDirectories(outputDir)
        model.save(outputDir, CheckpointFiles.MODEL_PREFIX)
        saveMetadata(outputDir, modelConfig, codec, trainingState)
    }

    fun saveMetadata(
        outputDir: Path,
        modelConfig: GptConfig,
        codec: TextCodec,
        trainingState: TrainingState,
    ) {
        Files.createDirectories(outputDir)
        val modelProperties = Properties().apply {
            setProperty("block_size", modelConfig.blockSize.toString())
            setProperty("vocab_size", modelConfig.vocabSize.toString())
            setProperty("n_layer", modelConfig.nLayer.toString())
            setProperty("n_head", modelConfig.nHead.toString())
            setProperty("n_embd", modelConfig.nEmbd.toString())
            setProperty("dropout", modelConfig.dropout.toString())
            setProperty("bias", modelConfig.bias.toString())
            setProperty("codec", codec.type)
        }
        val stateProperties = Properties().apply {
            setProperty("iter", trainingState.iter.toString())
            setProperty("best_val_loss", trainingState.bestValLoss.toString())
            setProperty("train_rng_state", trainingState.trainRngState.toString())
            setProperty("train_eval_rng_state", trainingState.trainEvalRngState.toString())
            setProperty("val_eval_rng_state", trainingState.valEvalRngState.toString())
        }
        PropertiesIO.store(
            outputDir.resolve(CheckpointFiles.MODEL_PROPERTIES),
            modelProperties,
            "nanoGPT Kotlin model config",
        )
        PropertiesIO.store(
            outputDir.resolve(CheckpointFiles.STATE_PROPERTIES),
            stateProperties,
            "nanoGPT Kotlin training state",
        )
        codec.copyArtifactsTo(outputDir)
    }

    private fun resolveExisting(basePath: Path, candidates: List<Path>): Path {
        candidates.firstOrNull(::hasMetadata)?.let { return it }
        error(
            buildString {
                append("No checkpoint metadata found for ")
                append(basePath)
                append(". Looked in: ")
                append(candidates.joinToString())
            },
        )
    }

    private fun hasMetadata(path: Path): Boolean {
        return Files.isRegularFile(path.resolve(CheckpointFiles.MODEL_PROPERTIES))
    }
}
