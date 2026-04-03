package dev.naoki.nanogpt

import ai.djl.Model
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import ai.djl.training.ParameterStore
import kotlin.io.path.Path
import kotlin.io.path.readText
import kotlin.random.Random

object SampleCli {
    @JvmStatic
    fun main(args: Array<String>) {
        if (Cli.wantsHelp(args)) {
            Cli.printUsage(
                listOf(
                    "Usage: sample [--key=value ...]",
                    "",
                    "Required:",
                    "  --checkpoint_dir=/path/to/out or /path/to/out/best",
                    "",
                    "Common options:",
                    "  --start='\\n'",
                    "  --start_file=/path/to/prompt.txt",
                    "  --num_samples=3",
                    "  --max_new_tokens=300",
                    "  --temperature=0.8",
                    "  --top_k=200",
                ),
            )
            return
        }

        val values = Cli.loadOverrides(args)
        val config = SampleConfig(
            checkpointDir = Cli.requirePath(values, "checkpoint_dir"),
            start = Cli.string(values, "start", "\n"),
            startFile = values["start_file"]?.let(::Path),
            numSamples = Cli.int(values, "num_samples", 10),
            maxNewTokens = Cli.int(values, "max_new_tokens", 500),
            temperature = Cli.float(values, "temperature", 0.8f),
            topK = Cli.int(values, "top_k", 200),
            seed = Cli.int(values, "seed", 1337),
        )

        val metadata = Checkpoints.loadMetadata(config.checkpointDir)
        val prompt = when {
            config.startFile != null -> config.startFile.readText()
            config.start.startsWith("FILE:") -> Path(config.start.removePrefix("FILE:")).readText()
            else -> config.start
        }

        Model.newInstance(CheckpointFiles.MODEL_PREFIX, "PyTorch").use { model ->
            model.block = GptModel(metadata.modelConfig)
            model.load(metadata.directory, CheckpointFiles.MODEL_PREFIX)
            NDManager.newBaseManager().use { manager ->
                val parameterStore = ParameterStore(manager, false)
                repeat(config.numSamples) {
                    val generated = generate(
                        model = model,
                        parameterStore = parameterStore,
                        manager = manager,
                        prompt = metadata.codec.encode(prompt),
                        maxNewTokens = config.maxNewTokens,
                        blockSize = metadata.modelConfig.blockSize,
                        temperature = config.temperature,
                        topK = config.topK,
                        random = Random(config.seed + it),
                    )
                    println(metadata.codec.decode(generated))
                    if (it != config.numSamples - 1) {
                        println("---------------")
                    }
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
}
