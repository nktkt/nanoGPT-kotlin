package dev.naoki.nanogpt

import ai.djl.Model
import ai.djl.ndarray.NDList
import ai.djl.ndarray.types.Shape
import ai.djl.nn.Parameter
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.Trainer
import ai.djl.training.initializer.Initializer
import ai.djl.training.initializer.NormalInitializer
import ai.djl.training.loss.SoftmaxCrossEntropyLoss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import ai.djl.training.tracker.WarmUpTracker
import java.nio.file.Files
import kotlin.io.path.Path
import kotlin.io.path.isDirectory
import kotlin.random.Random

object TrainCli {
    @JvmStatic
    fun main(args: Array<String>) {
        if (Cli.wantsHelp(args)) {
            Cli.printUsage(
                listOf(
                    "Usage: train [config.properties] [--key=value ...]",
                    "",
                    "Required:",
                    "  --dataset_dir=/path/to/dataset",
                    "",
                    "Common options:",
                    "  --out_dir=out-shakespeare-char",
                    "  --resume_from=/path/to/out/latest",
                    "  --device=cpu",
                    "  --batch_size=64",
                    "  --block_size=256",
                    "  --n_layer=6 --n_head=6 --n_embd=384",
                    "  --eval_interval=250 --eval_iters=50 --max_iters=5000",
                ),
            )
            return
        }

        val values = Cli.loadOverrides(args)
        val config = TrainConfig(
            datasetDir = Cli.requirePath(values, "dataset_dir"),
            outDir = values["out_dir"]?.let(::Path) ?: Path("out"),
            resumeFrom = values["resume_from"]?.let(::Path),
            evalInterval = Cli.int(values, "eval_interval", 250),
            logInterval = Cli.int(values, "log_interval", 10),
            evalIters = Cli.int(values, "eval_iters", 50),
            evalOnly = Cli.bool(values, "eval_only", false),
            alwaysSaveCheckpoint = Cli.bool(values, "always_save_checkpoint", false),
            gradientAccumulationSteps = Cli.int(values, "gradient_accumulation_steps", 1),
            batchSize = Cli.int(values, "batch_size", 64),
            blockSize = Cli.int(values, "block_size", 256),
            nLayer = Cli.int(values, "n_layer", 6),
            nHead = Cli.int(values, "n_head", 6),
            nEmbd = Cli.int(values, "n_embd", 384),
            dropout = Cli.float(values, "dropout", 0.2f),
            bias = Cli.bool(values, "bias", true),
            learningRate = Cli.float(values, "learning_rate", 1e-3f),
            maxIters = Cli.int(values, "max_iters", 5000),
            weightDecay = Cli.float(values, "weight_decay", 0.1f),
            beta1 = Cli.float(values, "beta1", 0.9f),
            beta2 = Cli.float(values, "beta2", 0.99f),
            gradClip = Cli.float(values, "grad_clip", 1.0f),
            warmupIters = Cli.int(values, "warmup_iters", 100),
            lrDecayIters = Cli.int(values, "lr_decay_iters", 5000),
            minLr = Cli.float(values, "min_lr", 1e-4f),
            device = Cli.string(values, "device", "cpu"),
            seed = Cli.int(values, "seed", 1337),
        )

        require(config.datasetDir.isDirectory()) { "dataset_dir does not exist: ${config.datasetDir}" }
        Files.createDirectories(config.outDir)

        TokenDataset(config.datasetDir).use { dataset ->
            val resumeDir = config.resumeFrom?.let(Checkpoints::resolveForResume)
            val resumeMetadata = resumeDir?.let(Checkpoints::loadMetadata)
            val datasetVocabSize = dataset.inferredVocabSize()
                ?: values["vocab_size"]?.toInt()
                ?: resumeMetadata?.modelConfig?.vocabSize
                ?: error("Could not infer vocab size. Provide --vocab_size or include vocab.txt in dataset_dir.")
            val modelConfig = resumeMetadata?.modelConfig ?: config.toModelConfig(datasetVocabSize)
            require(modelConfig.vocabSize == datasetVocabSize) {
                "Checkpoint vocab_size=${modelConfig.vocabSize} does not match dataset vocab_size=$datasetVocabSize"
            }

            val tokenLoss = SoftmaxCrossEntropyLoss("token_loss", 1f, 1, true, false)
            val optimizer = Optimizer.adamW()
                .optLearningRateTracker(buildTracker(config))
                .optBeta1(config.beta1)
                .optBeta2(config.beta2)
                .optWeightDecays(config.weightDecay)
                .optClipGrad(config.gradClip)
                .build()

            val trainingConfig = DefaultTrainingConfig(tokenLoss)
                .optDevices(arrayOf(config.deviceHandle()))
                .optOptimizer(optimizer)
                .optInitializer(NormalInitializer(0.02f), Parameter.Type.WEIGHT)
                .optInitializer(Initializer.ONES, Parameter.Type.GAMMA)
                .optInitializer(Initializer.ZEROS, Parameter.Type.BETA)
                .optInitializer(Initializer.ZEROS, Parameter.Type.BIAS)

            Model.newInstance(CheckpointFiles.MODEL_PREFIX, "PyTorch").use { model ->
                model.block = GptModel(modelConfig)
                model.newTrainer(trainingConfig).use { trainer ->
                    trainer.initialize(Shape(config.batchSize.toLong(), config.blockSize.toLong()))
                    if (resumeDir != null) {
                        model.load(resumeDir, CheckpointFiles.MODEL_PREFIX)
                        println("resumed_from=$resumeDir")
                        println("resume_note=optimizer state is reinitialized; weight resume only")
                    }

                    val codec = dataset.checkpointCodec()
                    val numParams = countParameters(model)
                    println("parameters=${"%.2f".format(numParams / 1_000_000.0)}M")

                    var completedIters = resumeMetadata?.trainingState?.iter ?: 0
                    var bestValLoss = resumeMetadata?.trainingState?.bestValLoss ?: Float.POSITIVE_INFINITY
                    val trainingRandom = Random(config.seed + completedIters)
                    val trainEvalRandom = Random(config.seed + 1)
                    val valEvalRandom = Random(config.seed + 2)
                    var lastLogTime = System.nanoTime()

                    Checkpoints.save(
                        model,
                        Checkpoints.latestDir(config.outDir),
                        modelConfig,
                        codec,
                        TrainingState(completedIters, bestValLoss),
                    )

                    while (true) {
                        if (completedIters % config.evalInterval == 0) {
                            val trainLoss = estimateLoss(trainer, tokenLoss, dataset, Split.TRAIN, config, trainEvalRandom)
                            val valLoss = estimateLoss(trainer, tokenLoss, dataset, Split.VAL, config, valEvalRandom)
                            if (valLoss < bestValLoss) {
                                bestValLoss = valLoss
                            }

                            val state = TrainingState(completedIters, bestValLoss)
                            Checkpoints.save(model, Checkpoints.latestDir(config.outDir), modelConfig, codec, state)
                            if (valLoss <= bestValLoss || config.alwaysSaveCheckpoint) {
                                Checkpoints.save(model, Checkpoints.bestDir(config.outDir), modelConfig, codec, state)
                            }
                            println(
                                "step $completedIters: train loss ${"%.4f".format(trainLoss)}, " +
                                    "val loss ${"%.4f".format(valLoss)}",
                            )
                            if (config.evalOnly) {
                                break
                            }
                        }

                        if (completedIters >= config.maxIters) {
                            break
                        }

                        val lossValue = trainIteration(trainer, tokenLoss, dataset, config, trainingRandom)
                        completedIters += 1
                        if (completedIters % config.logInterval == 0) {
                            val now = System.nanoTime()
                            val millis = (now - lastLogTime) / 1_000_000.0
                            lastLogTime = now
                            println("iter $completedIters: loss ${"%.4f".format(lossValue)}, time ${"%.2f".format(millis)}ms")
                        }
                    }

                    Checkpoints.save(
                        model,
                        Checkpoints.latestDir(config.outDir),
                        modelConfig,
                        codec,
                        TrainingState(completedIters, bestValLoss),
                    )
                }
            }
        }
    }

    private fun buildTracker(config: TrainConfig): Tracker {
        val cosine = Tracker.cosine()
            .setBaseValue(config.learningRate)
            .optFinalValue(config.minLr)
            .setMaxUpdates(config.lrDecayIters)
            .build()
        return if (config.warmupIters > 0) {
            Tracker.warmUp()
                .setMainTracker(cosine)
                .optWarmUpBeginValue(0f)
                .optWarmUpSteps(config.warmupIters)
                .optWarmUpMode(WarmUpTracker.Mode.LINEAR)
                .build()
        } else {
            cosine
        }
    }

    private fun trainIteration(
        trainer: Trainer,
        tokenLoss: SoftmaxCrossEntropyLoss,
        dataset: TokenDataset,
        config: TrainConfig,
        random: Random,
    ): Float {
        var totalLoss = 0f
        trainer.newGradientCollector().use { collector ->
            repeat(config.gradientAccumulationSteps) {
                dataset.sampleBatch(
                    split = Split.TRAIN,
                    rootManager = trainer.manager,
                    device = config.deviceHandle(),
                    batchSize = config.batchSize,
                    blockSize = config.blockSize,
                    random = random,
                ).use { batch ->
                    val predictions = trainer.forward(NDList(batch.data))
                    val lossValue = evaluateTokenLoss(tokenLoss, batch.labels, predictions)
                    totalLoss += lossValue.getFloat()
                    collector.backward(lossValue.div(config.gradientAccumulationSteps.toFloat()))
                }
            }
        }
        trainer.step()
        return totalLoss / config.gradientAccumulationSteps
    }

    private fun estimateLoss(
        trainer: Trainer,
        tokenLoss: SoftmaxCrossEntropyLoss,
        dataset: TokenDataset,
        split: Split,
        config: TrainConfig,
        random: Random,
    ): Float {
        var total = 0f
        repeat(config.evalIters) {
            dataset.sampleBatch(
                split = split,
                rootManager = trainer.manager,
                device = config.deviceHandle(),
                batchSize = config.batchSize,
                blockSize = config.blockSize,
                random = random,
            ).use { batch ->
                val predictions = trainer.evaluate(NDList(batch.data))
                total += evaluateTokenLoss(tokenLoss, batch.labels, predictions).getFloat()
            }
        }
        return total / config.evalIters
    }

    private fun evaluateTokenLoss(
        tokenLoss: SoftmaxCrossEntropyLoss,
        labels: ai.djl.ndarray.NDArray,
        predictions: NDList,
    ): ai.djl.ndarray.NDArray {
        val logProbs = predictions.singletonOrThrow()
        val vocabSize = logProbs.shape.get(logProbs.shape.dimension() - 1)
        val flatLogProbs = logProbs.reshape(-1, vocabSize)
        val flatLabels = labels.reshape(-1)
        return tokenLoss.evaluate(NDList(flatLabels), NDList(flatLogProbs))
    }

    private fun countParameters(model: Model): Long {
        return model.block.parameters.values().sumOf { parameter ->
            parameter.array?.size() ?: 0L
        }
    }
}
