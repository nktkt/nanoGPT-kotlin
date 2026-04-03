package dev.naoki.nanogpt

import kotlin.io.path.Path

object ImportGpt2Cli {
    @JvmStatic
    fun main(args: Array<String>) {
        if (Cli.wantsHelp(args)) {
            Cli.printUsage(
                listOf(
                    "Usage: import-gpt2 [--key=value ...]",
                    "",
                    "Required:",
                    "  --model_type=gpt2|gpt2-medium|gpt2-large|gpt2-xl",
                    "  --output_dir=/path/to/checkpoint",
                    "",
                    "Optional:",
                    "  --source_dir=/path/to/huggingface-model-dir",
                    "  --block_size=1024",
                    "  --dropout=0.0",
                ),
            )
            return
        }

        val values = Cli.loadOverrides(args)
        val config = ImportGpt2Config(
            modelType = Cli.string(values, "model_type", "gpt2"),
            sourceDir = values["source_dir"]?.let(::Path),
            outputDir = Cli.requirePath(values, "output_dir"),
            blockSize = Cli.int(values, "block_size", 1024),
            dropout = Cli.float(values, "dropout", 0.0f),
            seed = Cli.int(values, "seed", 1337),
        )

        Gpt2Importer.importCheckpoint(config)
        println("imported ${config.modelType} to ${config.outputDir}")
    }
}
