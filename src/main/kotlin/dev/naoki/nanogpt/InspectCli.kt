package dev.naoki.nanogpt

import java.nio.file.Files

object InspectCli {
    @JvmStatic
    fun main(args: Array<String>) {
        if (Cli.wantsHelp(args) || args.isEmpty()) {
            Cli.printUsage(
                listOf(
                    "Usage: inspect --checkpoint_dir=/path/to/out or /path/to/out/best",
                ),
            )
            return
        }

        val values = Cli.loadOverrides(args)
        val metadata = Checkpoints.loadMetadata(Cli.requirePath(values, "checkpoint_dir"))
        val state = metadata.trainingState
        println("checkpoint_dir=${metadata.directory}")
        println("codec=${metadata.codec.type}")
        println("block_size=${metadata.modelConfig.blockSize}")
        println("vocab_size=${metadata.modelConfig.vocabSize}")
        println("n_layer=${metadata.modelConfig.nLayer}")
        println("n_head=${metadata.modelConfig.nHead}")
        println("n_embd=${metadata.modelConfig.nEmbd}")
        println("dropout=${metadata.modelConfig.dropout}")
        println("bias=${metadata.modelConfig.bias}")
        println(
            "optimizer_state=" +
                Files.exists(
                    metadata.directory
                        .resolve(CheckpointFiles.OPTIMIZER_DIR)
                        .resolve(CheckpointFiles.OPTIMIZER_MANIFEST),
                ),
        )
        if (state != null) {
            println("iter=${state.iter}")
            println("best_val_loss=${state.bestValLoss}")
        }
    }
}
