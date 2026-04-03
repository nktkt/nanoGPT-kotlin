package dev.naoki.nanogpt

import ai.djl.Device
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.Shape
import java.io.Closeable
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.nio.file.Path
import java.nio.file.StandardOpenOption
import kotlin.io.path.exists
import kotlin.io.path.readText
import kotlin.math.min

private const val SEGMENT_BYTES: Long = 1L shl 30

class UInt16MemMap(path: Path) : Closeable {
    private val channel = FileChannel.open(path, StandardOpenOption.READ)
    private val segments: List<MappedByteBuffer>
    val tokenCount: Long

    init {
        val sizeBytes = channel.size()
        require(sizeBytes % 2L == 0L) { "Expected uint16 file: $path" }
        tokenCount = sizeBytes / 2L
        val mapped = mutableListOf<MappedByteBuffer>()
        var offset = 0L
        while (offset < sizeBytes) {
            val length = min(SEGMENT_BYTES, sizeBytes - offset)
            val buffer = channel.map(FileChannel.MapMode.READ_ONLY, offset, length)
            buffer.order(ByteOrder.LITTLE_ENDIAN)
            mapped += buffer
            offset += length
        }
        segments = mapped
    }

    fun get(index: Long): Int {
        require(index in 0 until tokenCount) { "Token index out of range: $index" }
        val byteIndex = index * 2L
        val segmentIndex = (byteIndex / SEGMENT_BYTES).toInt()
        val localIndex = (byteIndex % SEGMENT_BYTES).toInt()
        return segments[segmentIndex].getShort(localIndex).toInt() and 0xffff
    }

    override fun close() {
        channel.close()
    }
}

class TokenBatch(
    private val manager: NDManager,
    val data: NDArray,
    val labels: NDArray,
) : Closeable {
    override fun close() {
        manager.close()
    }
}

class TokenDataset(private val datasetDir: Path) : Closeable {
    private val train = UInt16MemMap(datasetDir.resolve("train.bin"))
    private val validation = UInt16MemMap(datasetDir.resolve("val.bin"))
    val characterCodec: CharacterCodec? = CharacterCodec.maybeLoad(datasetDir)

    fun inferredVocabSize(): Int? = characterCodec?.vocabSize

    fun checkpointCodec(): TextCodec = characterCodec ?: Gpt2Codec()

    fun sampleBatch(
        split: Split,
        rootManager: NDManager,
        device: Device,
        batchSize: Int,
        blockSize: Int,
        random: StatefulRandom,
    ): TokenBatch {
        val source = if (split == Split.TRAIN) train else validation
        require(source.tokenCount > blockSize.toLong()) {
            "Dataset ${split.name.lowercase()} is too small for block_size=$blockSize"
        }
        val batchManager = rootManager.newSubManager()
        val tokens = LongArray(batchSize * blockSize)
        val labels = LongArray(batchSize * blockSize)
        repeat(batchSize) { batch ->
            val start = random.nextLong(source.tokenCount - blockSize.toLong())
            val base = batch * blockSize
            repeat(blockSize) { offset ->
                tokens[base + offset] = source.get(start + offset).toLong()
                labels[base + offset] = source.get(start + offset + 1).toLong()
            }
        }
        val x = batchManager.create(tokens, Shape(batchSize.toLong(), blockSize.toLong())).toDevice(device, true)
        val y = batchManager.create(labels, Shape(batchSize.toLong(), blockSize.toLong())).toDevice(device, true)
        return TokenBatch(batchManager, x, y)
    }

    override fun close() {
        train.close()
        validation.close()
    }
}

enum class Split {
    TRAIN,
    VAL,
}
