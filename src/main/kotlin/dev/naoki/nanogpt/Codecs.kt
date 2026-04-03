package dev.naoki.nanogpt

import com.knuddels.jtokkit.Encodings
import com.knuddels.jtokkit.api.EncodingType
import com.knuddels.jtokkit.api.IntArrayList
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path
import kotlin.io.path.readLines

interface TextCodec {
    fun encode(text: String): IntArray
    fun decode(tokens: IntArray): String
    fun copyArtifactsTo(outputDir: Path) {}
    val type: String
}

class CharacterCodec private constructor(
    private val codePoints: IntArray,
) : TextCodec {
    private val stoi = codePoints.withIndex().associate { it.value to it.index }

    override val type: String = "char"

    override fun encode(text: String): IntArray {
        val cps = text.codePoints().toArray()
        return IntArray(cps.size) { index ->
            stoi[cps[index]] ?: error("Character U+${cps[index].toString(16)} is not in vocab.txt")
        }
    }

    override fun decode(tokens: IntArray): String {
        val builder = StringBuilder()
        tokens.forEach { token ->
            builder.append(String(Character.toChars(codePoints[token])))
        }
        return builder.toString()
    }

    override fun copyArtifactsTo(outputDir: Path) {
        Files.createDirectories(outputDir)
        val target = outputDir.resolve(CheckpointFiles.VOCAB_FILE)
        Files.write(
            target,
            codePoints.map(Int::toString),
            StandardCharsets.UTF_8,
        )
    }

    val vocabSize: Int
        get() = codePoints.size

    companion object {
        fun fromVocabularyFile(path: Path): CharacterCodec {
            val codePoints = path.readLines().filter { it.isNotBlank() }.map(String::toInt).toIntArray()
            return CharacterCodec(codePoints)
        }

        fun fromText(text: String): CharacterCodec {
            val uniqueSorted = text.codePoints().toArray().toSortedSet().toIntArray()
            return CharacterCodec(uniqueSorted)
        }

        fun maybeLoad(path: Path): CharacterCodec? {
            val vocabPath = path.resolve(CheckpointFiles.VOCAB_FILE)
            return if (Files.exists(vocabPath)) fromVocabularyFile(vocabPath) else null
        }
    }
}

class Gpt2Codec : TextCodec {
    private val registry = Encodings.newDefaultEncodingRegistry()
    private val encoding = registry.getEncoding(EncodingType.R50K_BASE)

    override val type: String = "gpt2"

    override fun encode(text: String): IntArray {
        return encoding.encode(text).toArray()
    }

    override fun decode(tokens: IntArray): String {
        val list = IntArrayList(tokens.size)
        tokens.forEach(list::add)
        return encoding.decode(list)
    }
}
