package dev.naoki.nanogpt

object NanoGptCli {
    @JvmStatic
    fun main(args: Array<String>) {
        if (args.isEmpty() || args.first() in setOf("-h", "--help", "help")) {
            Cli.printUsage(
                listOf(
                    "nanoGPT Kotlin",
                    "",
                    "Usage:",
                    "  nanogpt <command> [options]",
                    "",
                    "Commands:",
                    "  prepare   Build train.bin, val.bin, and vocab.txt from plain text",
                    "  train     Train a GPT model from scratch or resume from weights",
                    "  sample    Generate text from a checkpoint",
                    "  inspect   Print checkpoint metadata",
                    "  import-gpt2  Import Hugging Face GPT-2 weights into a nanoGPT Kotlin checkpoint",
                    "",
                    "Examples:",
                    "  nanogpt prepare --input=data/input.txt --output_dir=data/shakespeare_char",
                    "  nanogpt train configs/train-shakespeare-char.properties",
                    "  nanogpt sample --checkpoint_dir=out-shakespeare-char --start='\\n'",
                ),
            )
            return
        }

        val command = args.first()
        val remaining = args.copyOfRange(1, args.size)
        when (command) {
            "prepare" -> PrepareTextCli.main(remaining)
            "train" -> TrainCli.main(remaining)
            "sample" -> SampleCli.main(remaining)
            "inspect" -> InspectCli.main(remaining)
            "import-gpt2" -> ImportGpt2Cli.main(remaining)
            else -> error("Unknown command '$command'. Use --help to see available commands.")
        }
    }
}
