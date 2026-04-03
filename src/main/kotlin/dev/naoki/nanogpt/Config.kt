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
)

data class TrainConfig(
    val datasetDir: Path,
    val outDir: Path,
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
    fun deviceHandle(): Device {
        return when {
            device.equals("cpu", ignoreCase = true) -> Device.cpu()
            device.equals("gpu", ignoreCase = true) -> Device.gpu()
            device.startsWith("gpu:", ignoreCase = true) -> Device.gpu(device.substringAfter(':').toInt())
            else -> Device.cpu()
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
    val numSamples: Int = 10,
    val maxNewTokens: Int = 500,
    val temperature: Float = 0.8f,
    val topK: Int = 200,
    val seed: Int = 1337,
)

data class PrepareConfig(
    val input: Path,
    val outputDir: Path,
    val trainSplit: Float = 0.9f,
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
}
