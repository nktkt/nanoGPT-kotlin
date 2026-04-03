package dev.naoki.nanogpt

import ai.djl.Model
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Parameter
import com.google.gson.JsonObject
import com.google.gson.JsonParser
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardOpenOption
import kotlin.io.path.Path
import kotlin.io.path.exists

object Gpt2Importer {
    fun modelConfig(modelType: String, blockSize: Int = 1024, dropout: Float = 0.0f): GptConfig {
        val spec = spec(modelType)
        return GptConfig(
            blockSize = blockSize,
            vocabSize = 50257,
            nLayer = spec.nLayer,
            nHead = spec.nHead,
            nEmbd = spec.nEmbd,
            dropout = dropout,
            bias = true,
        )
    }

    fun importCheckpoint(config: ImportGpt2Config) {
        val modelConfig = modelConfig(config.modelType, config.blockSize, config.dropout)
        Files.createDirectories(config.outputDir)
        val source = HfGpt2Source.open(config.modelType, config.sourceDir, config.outputDir.resolve(".hf-cache"))
        Model.newInstance(CheckpointFiles.MODEL_PREFIX, "PyTorch").use { model ->
            val gptModel = GptModel(modelConfig)
            model.block = gptModel
            model.block.initialize(model.ndManager, DataType.FLOAT32, Shape(1, modelConfig.blockSize.toLong()))
            initializePretrained(model, model.ndManager, config.modelType, config.dropout, source)
            Checkpoints.save(
                model = model,
                outputDir = config.outputDir,
                modelConfig = modelConfig,
                codec = Gpt2Codec(),
                trainingState = TrainingState(
                    iter = 0,
                    bestValLoss = Float.POSITIVE_INFINITY,
                    trainRngState = config.seed.toLong(),
                    trainEvalRngState = (config.seed + 1).toLong(),
                    valEvalRngState = (config.seed + 2).toLong(),
                ),
            )
        }
    }

    fun initializePretrained(
        model: Model,
        manager: NDManager,
        modelType: String,
        dropout: Float,
        sourceDir: Path? = null,
    ) {
        val source = HfGpt2Source.open(modelType, sourceDir, Path("/tmp/nanogpt-kotlin-hf-cache"))
        initializePretrained(model, manager, modelType, dropout, source)
    }

    private fun initializePretrained(
        model: Model,
        manager: NDManager,
        modelType: String,
        dropout: Float,
        source: HfGpt2Source,
    ) {
        val gptModel = model.block as? GptModel ?: error("Model block must be GptModel")
        val targetConfig = gptModel.config()
        val expected = modelConfig(modelType, targetConfig.blockSize, dropout)
        require(targetConfig.nLayer == expected.nLayer) { "n_layer mismatch for $modelType" }
        require(targetConfig.nHead == expected.nHead) { "n_head mismatch for $modelType" }
        require(targetConfig.nEmbd == expected.nEmbd) { "n_embd mismatch for $modelType" }
        require(targetConfig.vocabSize == expected.vocabSize) { "vocab_size mismatch for $modelType" }

        source.use {
            assign(gptModel.tokenEmbeddingWeight(), source.tensor("transformer.wte.weight", manager))

            val positionEmbedding = source.tensor("transformer.wpe.weight", manager)
            val croppedPositionEmbedding = if (targetConfig.blockSize < 1024) {
                positionEmbedding.get(NDIndex("0:${targetConfig.blockSize}, :"))
            } else {
                positionEmbedding
            }
            assign(gptModel.positionEmbeddingWeight(), croppedPositionEmbedding)

            repeat(targetConfig.nLayer) { index ->
                val block = gptModel.transformerBlock(index)
                assign(block.ln1Weight(), source.tensor("transformer.h.$index.ln_1.weight", manager))
                block.ln1Bias()?.let { assign(it, source.tensor("transformer.h.$index.ln_1.bias", manager)) }
                assign(block.ln2Weight(), source.tensor("transformer.h.$index.ln_2.weight", manager))
                block.ln2Bias()?.let { assign(it, source.tensor("transformer.h.$index.ln_2.bias", manager)) }

                val attn = block.attention()
                assign(attn.qkvWeight(), source.tensor("transformer.h.$index.attn.c_attn.weight", manager).transpose())
                attn.qkvBias()?.let { assign(it, source.tensor("transformer.h.$index.attn.c_attn.bias", manager)) }
                assign(attn.projWeight(), source.tensor("transformer.h.$index.attn.c_proj.weight", manager).transpose())
                attn.projBias()?.let { assign(it, source.tensor("transformer.h.$index.attn.c_proj.bias", manager)) }

                val mlp = block.mlpBlock()
                assign(mlp.fcWeight(), source.tensor("transformer.h.$index.mlp.c_fc.weight", manager).transpose())
                mlp.fcBias()?.let { assign(it, source.tensor("transformer.h.$index.mlp.c_fc.bias", manager)) }
                assign(mlp.projWeight(), source.tensor("transformer.h.$index.mlp.c_proj.weight", manager).transpose())
                mlp.projBias()?.let { assign(it, source.tensor("transformer.h.$index.mlp.c_proj.bias", manager)) }
            }

            assign(gptModel.finalNormWeight(), source.tensor("transformer.ln_f.weight", manager))
            gptModel.finalNormBias()?.let { assign(it, source.tensor("transformer.ln_f.bias", manager)) }
        }
    }

    private fun assign(parameter: Parameter, source: NDArray) {
        val target = parameter.array
        require(source.shape == target.shape) { "Shape mismatch for ${parameter.name}: ${source.shape} != ${target.shape}" }
        source.toDevice(target.device, true).toType(target.dataType, false).copyTo(target)
    }

    private fun spec(modelType: String): Gpt2Spec {
        return when (modelType) {
            "gpt2" -> Gpt2Spec(12, 12, 768)
            "gpt2-medium" -> Gpt2Spec(24, 16, 1024)
            "gpt2-large" -> Gpt2Spec(36, 20, 1280)
            "gpt2-xl" -> Gpt2Spec(48, 25, 1600)
            else -> error("Unsupported GPT-2 model type: $modelType")
        }
    }

    private data class Gpt2Spec(
        val nLayer: Int,
        val nHead: Int,
        val nEmbd: Int,
    )
}

private class HfGpt2Source private constructor(
    private val tensorFiles: Map<String, SafeTensorFile>,
) : AutoCloseable {
    fun tensor(name: String, manager: NDManager): NDArray {
        val file = tensorFiles[name] ?: error("Missing tensor '$name' in safetensors source")
        return file.tensor(name, manager)
    }

    override fun close() {
        tensorFiles.values.toSet().forEach(SafeTensorFile::close)
    }

    companion object {
        fun open(modelType: String, sourceDir: Path?, cacheDir: Path): HfGpt2Source {
            val root = sourceDir ?: downloadToCache(modelType, cacheDir)
            val indexPath = root.resolve("model.safetensors.index.json")
            return if (indexPath.exists()) {
                fromIndex(root, indexPath)
            } else {
                val tensorPath = root.resolve("model.safetensors")
                require(Files.exists(tensorPath)) { "Could not find model.safetensors in $root" }
                val file = SafeTensorFile.open(tensorPath)
                val mapping = file.tensorNames().associateWith { file }
                HfGpt2Source(mapping)
            }
        }

        private fun fromIndex(root: Path, indexPath: Path): HfGpt2Source {
            val json = JsonParser.parseString(Files.readString(indexPath)).asJsonObject
            val weightMap = json.getAsJsonObject("weight_map")
            val filesByName = linkedMapOf<String, SafeTensorFile>()
            val tensors = linkedMapOf<String, SafeTensorFile>()
            weightMap.entrySet().forEach { (tensorName, fileNameElement) ->
                val fileName = fileNameElement.asString
                val file = filesByName.getOrPut(fileName) { SafeTensorFile.open(root.resolve(fileName)) }
                tensors[tensorName] = file
            }
            return HfGpt2Source(tensors)
        }

        private fun downloadToCache(modelType: String, cacheDir: Path): Path {
            val targetDir = cacheDir.resolve(modelType)
            Files.createDirectories(targetDir)
            val baseUrl = "https://huggingface.co/$modelType/resolve/main"
            downloadIfMissing("$baseUrl/config.json", targetDir.resolve("config.json"))
            val indexPath = targetDir.resolve("model.safetensors.index.json")
            val tensorPath = targetDir.resolve("model.safetensors")
            if (!Files.exists(indexPath) && !Files.exists(tensorPath)) {
                runCatching {
                    downloadIfMissing("$baseUrl/model.safetensors.index.json", indexPath)
                }.onFailure {
                    downloadIfMissing("$baseUrl/model.safetensors", tensorPath)
                }
            }
            if (Files.exists(indexPath)) {
                val json = JsonParser.parseString(Files.readString(indexPath)).asJsonObject
                val weightMap = json.getAsJsonObject("weight_map")
                weightMap.entrySet()
                    .map { it.value.asString }
                    .toSet()
                    .forEach { shard ->
                        downloadIfMissing("$baseUrl/$shard", targetDir.resolve(shard))
                    }
            } else {
                downloadIfMissing("$baseUrl/model.safetensors", tensorPath)
            }
            return targetDir
        }

        private fun downloadIfMissing(url: String, path: Path) {
            if (Files.exists(path)) {
                return
            }
            Files.createDirectories(path.parent)
            java.net.URI(url).toURL().openStream().use { input ->
                Files.copy(input, path)
            }
        }
    }
}

private class SafeTensorFile private constructor(
    private val channel: FileChannel,
    private val buffer: ByteBuffer,
    private val tensors: Map<String, SafeTensorEntry>,
) : AutoCloseable {
    fun tensorNames(): Set<String> = tensors.keys

    fun tensor(name: String, manager: NDManager): NDArray {
        val entry = tensors[name] ?: error("Tensor '$name' not found")
        return manager.create(entry.readFloatArray(buffer), entry.shape)
    }

    override fun close() {
        channel.close()
    }

    companion object {
        fun open(path: Path): SafeTensorFile {
            val channel = FileChannel.open(path, StandardOpenOption.READ)
            val buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, channel.size()).order(ByteOrder.LITTLE_ENDIAN)
            val headerLength = buffer.getLong(0).toInt()
            val headerBytes = ByteArray(headerLength)
            val headerStart = 8
            val headerSlice = buffer.duplicate()
            headerSlice.position(headerStart)
            headerSlice.get(headerBytes)
            val headerJson = JsonParser.parseString(String(headerBytes)).asJsonObject
            val entries = linkedMapOf<String, SafeTensorEntry>()
            headerJson.entrySet().forEach { (name, value) ->
                if (name == "__metadata__") {
                    return@forEach
                }
                val tensor = value.asJsonObject
                val dtype = tensor.get("dtype").asString
                val shape = Shape(*tensor.getAsJsonArray("shape").map { it.asLong }.toLongArray())
                val offsets = tensor.getAsJsonArray("data_offsets")
                val dataStart = headerStart + headerLength + offsets[0].asInt
                val dataEnd = headerStart + headerLength + offsets[1].asInt
                entries[name] = SafeTensorEntry(dtype, shape, dataStart, dataEnd)
            }
            return SafeTensorFile(channel, buffer, entries)
        }
    }
}

private data class SafeTensorEntry(
    val dtype: String,
    val shape: Shape,
    val start: Int,
    val end: Int,
) {
    fun readFloatArray(buffer: ByteBuffer): FloatArray {
        val slice = buffer.duplicate().order(ByteOrder.LITTLE_ENDIAN)
        slice.position(start)
        slice.limit(end)
        return when (dtype) {
            "F32" -> {
                val view = slice.slice().order(ByteOrder.LITTLE_ENDIAN)
                FloatArray(shape.size().toInt()) { view.float }
            }
            "F16" -> {
                val view = slice.slice().order(ByteOrder.LITTLE_ENDIAN)
                FloatArray(shape.size().toInt()) { halfToFloat(view.short.toInt() and 0xffff) }
            }
            "BF16" -> {
                val view = slice.slice().order(ByteOrder.LITTLE_ENDIAN)
                FloatArray(shape.size().toInt()) { java.lang.Float.intBitsToFloat((view.short.toInt() and 0xffff) shl 16) }
            }
            else -> error("Unsupported safetensors dtype: $dtype")
        }
    }

    private fun halfToFloat(bits: Int): Float {
        val sign = (bits ushr 15) and 0x1
        val exponent = (bits ushr 10) and 0x1f
        val mantissa = bits and 0x3ff
        val value = when {
            exponent == 0 -> {
                if (mantissa == 0) {
                    sign shl 31
                } else {
                    var exp = -14
                    var man = mantissa
                    while ((man and 0x400) == 0) {
                        man = man shl 1
                        exp -= 1
                    }
                    man = man and 0x3ff
                    (sign shl 31) or ((exp + 127) shl 23) or (man shl 13)
                }
            }
            exponent == 0x1f -> (sign shl 31) or 0x7f800000 or (mantissa shl 13)
            else -> (sign shl 31) or (((exponent - 15 + 127) and 0xff) shl 23) or (mantissa shl 13)
        }
        return java.lang.Float.intBitsToFloat(value)
    }
}
