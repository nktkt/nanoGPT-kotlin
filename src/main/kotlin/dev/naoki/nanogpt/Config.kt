package dev.naoki.nanogpt

import ai.djl.Device
import java.io.InputStream
import java.io.OutputStream
import java.nio.file.Files
import java.nio.file.Path
import java.util.Properties
import kotlin.io.path.inputStream
import kotlin.io.path.outputStream

data class GptConfig(
    val blockSize: Int = 256,
    val vocabSize: Int,
    val nLayer: Int = 6,
    val nHead: Int = 6,
    val nEmbd: Int = 384,
    val dropout: Float = 0.2f,
    val bias: Boolean = true,
) {
    init {
        require(blockSize > 0) { "block_size must be positive" }
        require(vocabSize > 0) { "vocab_size must be positive" }
        require(nLayer > 0) { "n_layer must be positive" }
        require(nHead > 0) { "n_head must be positive" }
        require(nEmbd > 0) { "n_embd must be positive" }
        require(nEmbd % nHead == 0) { "n_embd must be divisible by n_head" }
        require(dropout in 0f..1f) { "dropout must be between 0 and 1" }
    }
}

data class TrainConfig(
    val datasetDir: Path,
    val outDir: Path,
    val resumeFrom: Path? = null,
    val evalInterval: Int = 250,
    val logInterval: Int = 10,
    val evalIters: Int = 50,
    val evalOnly: Boolean = false,
    val alwaysSaveCheckpoint: Boolean = false,
    val gradientAccumulationSteps: Int = 1,
    val batchSize: Int = 64,
    val blockSize: Int = 256,
    val nLayer: Int = 6,
    val nHead: Int = 6,
    val nEmbd: Int = 384,
    val dropout: Float = 0.2f,
    val bias: Boolean = true,
    val learningRate: Float = 1e-3f,
    val maxIters: Int = 5000,
    val weightDecay: Float = 0.1f,
    val beta1: Float = 0.9f,
    val beta2: Float = 0.99f,
    val gradClip: Float = 1.0f,
    val warmupIters: Int = 100,
    val lrDecayIters: Int = 5000,
    val minLr: Float = 1e-4f,
    val device: String = "cpu",
    val seed: Int = 1337,
) {
    init {
        require(evalInterval > 0) { "eval_interval must be positive" }
        require(logInterval > 0) { "log_interval must be positive" }
        require(evalIters > 0) { "eval_iters must be positive" }
        require(gradientAccumulationSteps > 0) { "gradient_accumulation_steps must be positive" }
        require(batchSize > 0) { "batch_size must be positive" }
        require(blockSize > 0) { "block_size must be positive" }
        require(nLayer > 0) { "n_layer must be positive" }
        require(nHead > 0) { "n_head must be positive" }
        require(nEmbd > 0) { "n_embd must be positive" }
        require(nEmbd % nHead == 0) { "n_embd must be divisible by n_head" }
        require(dropout in 0f..1f) { "dropout must be between 0 and 1" }
        require(learningRate > 0f) { "learning_rate must be positive" }
        require(maxIters >= 0) { "max_iters must be non-negative" }
        require(weightDecay >= 0f) { "weight_decay must be non-negative" }
        require(beta1 in 0f..1f) { "beta1 must be between 0 and 1" }
        require(beta2 in 0f..1f) { "beta2 must be between 0 and 1" }
        require(gradClip > 0f) { "grad_clip must be positive" }
        require(warmupIters >= 0) { "warmup_iters must be non-negative" }
        require(lrDecayIters > 0) { "lr_decay_iters must be positive" }
        require(minLr >= 0f) { "min_lr must be non-negative" }
    }

    fun deviceHandle(): Device {
        return when {
            device.equals("cpu", ignoreCase = true) -> Device.cpu()
            device.equals("gpu", ignoreCase = true) -> Device.gpu()
            device.startsWith("gpu:", ignoreCase = true) -> Device.gpu(device.substringAfter(':').toInt())
            else -> error("Unsupported device '$device'. Use cpu, gpu, or gpu:<index>.")
        }
    }

    fun toModelConfig(vocabSize: Int): GptConfig {
        return GptConfig(
            blockSize = blockSize,
            vocabSize = vocabSize,
            nLayer = nLayer,
            nHead = nHead,
            nEmbd = nEmbd,
            dropout = dropout,
            bias = bias,
        )
    }
}

data class SampleConfig(
    val checkpointDir: Path,
    val start: String = "\n",
    val startFile: Path? = null,
    val numSamples: Int = 10,
    val maxNewTokens: Int = 500,
    val temperature: Float = 0.8f,
    val topK: Int = 200,
    val seed: Int = 1337,
) {
    init {
        require(numSamples > 0) { "num_samples must be positive" }
        require(maxNewTokens >= 0) { "max_new_tokens must be non-negative" }
        require(temperature > 0f) { "temperature must be positive" }
        require(topK >= 0) { "top_k must be non-negative" }
    }
}

data class PrepareConfig(
    val input: Path,
    val outputDir: Path,
    val trainSplit: Float = 0.9f,
) {
    init {
        require(trainSplit > 0f && trainSplit < 1f) { "train_split must be between 0 and 1" }
    }
}

data class TrainingState(
    val iter: Int,
    val bestValLoss: Float,
)

object PropertiesIO {
    fun load(path: Path): Properties {
        val properties = Properties()
        path.inputStream().use(properties::load)
        return properties
    }

    fun store(path: Path, properties: Properties, comment: String) {
        Files.createDirectories(path.parent)
        path.outputStream().use { output -> properties.store(output, comment) }
    }
}

object CheckpointFiles {
    const val MODEL_PREFIX = "nanogpt-kotlin"
    const val MODEL_PROPERTIES = "model.properties"
    const val STATE_PROPERTIES = "state.properties"
    const val VOCAB_FILE = "vocab.txt"
    const val BEST_DIR = "best"
    const val LATEST_DIR = "latest"
}
