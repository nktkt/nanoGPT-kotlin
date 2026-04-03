package dev.naoki.nanogpt

import ai.djl.Model
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.Shape
import ai.djl.training.ParameterStore
import java.nio.file.Path
import kotlin.io.path.Path
import kotlin.random.Random

object SampleCli {
    @JvmStatic
    fun main(args: Array<String>) {
        val values = Cli.loadOverrides(args)
        val checkpointDir = Cli.requirePath(values, "checkpoint_dir")
        val config = SampleConfig(
            checkpointDir = checkpointDir,
            start = Cli.string(values, "start", "\n"),
            numSamples = Cli.int(values, "num_samples", 10),
            maxNewTokens = Cli.int(values, "max_new_tokens", 500),
            temperature = Cli.float(values, "temperature", 0.8f),
            topK = Cli.int(values, "top_k", 200),
            seed = Cli.int(values, "seed", 1337),
        )

        val modelConfig = loadModelConfig(checkpointDir)
        val codec = CharacterCodec.maybeLoad(checkpointDir) ?: Gpt2Codec()
        val prompt = if (config.start.startsWith("FILE:")) {
            Path(config.start.removePrefix("FILE:")).toFile().readText()
        } else {
            config.start
        }

        Model.newInstance(CheckpointFiles.MODEL_PREFIX, "PyTorch").use { model ->
            model.block = GptModel(modelConfig)
            model.load(checkpointDir, CheckpointFiles.MODEL_PREFIX)
            NDManager.newBaseManager().use { manager ->
                val parameterStore = ParameterStore(manager, false)
                repeat(config.numSamples) {
                    val generated = generate(
                        model = model,
                        parameterStore = parameterStore,
                        manager = manager,
                        prompt = codec.encode(prompt),
                        maxNewTokens = config.maxNewTokens,
                        blockSize = modelConfig.blockSize,
                        temperature = config.temperature,
                        topK = config.topK,
                        random = Random(config.seed + it),
                    )
                    println(codec.decode(generated))
                    println("---------------")
                }
            }
        }
    }

    private fun generate(
        model: Model,
        parameterStore: ParameterStore,
        manager: NDManager,
        prompt: IntArray,
        maxNewTokens: Int,
        blockSize: Int,
        temperature: Float,
        topK: Int,
        random: Random,
    ): IntArray {
        val tokens = prompt.toMutableList()
        repeat(maxNewTokens) {
            val context = tokens.takeLast(blockSize).toIntArray()
            val input = manager.create(context.map(Int::toLong).toLongArray(), Shape(1, context.size.toLong()))
            val output = model.block.forward(parameterStore, NDList(input), false).singletonOrThrow()
            val last = output.get(":, -1, :").reshape(output.shape.get(2)).toFloatArray()
            val nextToken = sampleFromLogProbs(last, temperature, topK, random)
            tokens += nextToken
        }
        return tokens.toIntArray()
    }

    private fun sampleFromLogProbs(logProbs: FloatArray, temperature: Float, topK: Int, random: Random): Int {
        val adjusted = FloatArray(logProbs.size) { index -> logProbs[index] / temperature.coerceAtLeast(1e-6f) }
        if (topK in 1 until adjusted.size) {
            val threshold = adjusted.sortedDescending()[topK - 1]
            adjusted.indices.forEach { index ->
                if (adjusted[index] < threshold) {
                    adjusted[index] = Float.NEGATIVE_INFINITY
                }
            }
        }
        val max = adjusted.filter { it.isFinite() }.maxOrNull() ?: 0f
        val weights = DoubleArray(adjusted.size) { index ->
            if (adjusted[index].isFinite()) kotlin.math.exp((adjusted[index] - max).toDouble()) else 0.0
        }
        val total = weights.sum()
        var threshold = random.nextDouble() * total
        weights.forEachIndexed { index, value ->
            threshold -= value
            if (threshold <= 0.0) {
                return index
            }
        }
        return weights.indices.maxByOrNull { weights[it] } ?: 0
    }

    private fun loadModelConfig(checkpointDir: Path): GptConfig {
        val props = PropertiesIO.load(checkpointDir.resolve(CheckpointFiles.MODEL_PROPERTIES))
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
}
