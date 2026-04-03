package dev.naoki.nanogpt

import java.io.BufferedOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.Path
import kotlin.io.path.readText

object PrepareTextCli {
    @JvmStatic
    fun main(args: Array<String>) {
        val values = Cli.loadOverrides(args)
        val config = PrepareConfig(
            input = Cli.requirePath(values, "input"),
            outputDir = values["output_dir"]?.let(::Path) ?: Path("data/text-char"),
            trainSplit = Cli.float(values, "train_split", 0.9f),
        )

        val text = config.input.readText(StandardCharsets.UTF_8)
        val codec = CharacterCodec.fromText(text)
        val encoded = codec.encode(text)
        require(codec.vocabSize <= 65535) { "Character vocab exceeds uint16 capacity: ${codec.vocabSize}" }

        val splitIndex = (encoded.size * config.trainSplit).toInt().coerceIn(1, encoded.size - 1)
        val train = encoded.copyOfRange(0, splitIndex)
        val validation = encoded.copyOfRange(splitIndex, encoded.size)

        Files.createDirectories(config.outputDir)
        writeUInt16Bin(config.outputDir.resolve("train.bin"), train)
        writeUInt16Bin(config.outputDir.resolve("val.bin"), validation)
        codec.copyArtifactsTo(config.outputDir)

        println("prepared ${config.outputDir}")
        println("train_tokens=${train.size}")
        println("val_tokens=${validation.size}")
        println("vocab_size=${codec.vocabSize}")
    }

    private fun writeUInt16Bin(path: Path, values: IntArray) {
        BufferedOutputStream(Files.newOutputStream(path)).use { output ->
            val buffer = ByteBuffer.allocate(8192).order(ByteOrder.LITTLE_ENDIAN)
            values.forEach { value ->
                if (buffer.remaining() < 2) {
                    output.write(buffer.flip().let { it.array().copyOf(it.limit()) })
                    buffer.clear()
                }
                buffer.putShort(value.toShort())
            }
            if (buffer.position() > 0) {
                output.write(buffer.flip().let { it.array().copyOf(it.limit()) })
            }
        }
    }
}
