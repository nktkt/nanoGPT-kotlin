package dev.naoki.nanogpt

import ai.djl.Model
import ai.djl.engine.Engine
import ai.djl.ndarray.NDArray
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
import ai.djl.util.RandomUtils
import java.nio.file.Files
import kotlin.io.path.Path
import kotlin.io.path.isDirectory

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
                    "  --init_from=scratch|resume|gpt2|gpt2-medium|gpt2-large|gpt2-xl",
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
            initFrom = Cli.string(values, "init_from", if (values.containsKey("resume_from")) "resume" else "scratch"),
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
            decayLr = Cli.bool(values, "decay_lr", true),
            warmupIters = Cli.int(values, "warmup_iters", 100),
            lrDecayIters = Cli.int(values, "lr_decay_iters", 5000),
            minLr = Cli.float(values, "min_lr", 1e-4f),
            device = Cli.string(values, "device", "cpu"),
            seed = Cli.int(values, "seed", 1337),
        )

        require(config.datasetDir.isDirectory()) { "dataset_dir does not exist: ${config.datasetDir}" }
        Files.createDirectories(config.outDir)
        RandomUtils.RANDOM.setSeed(config.seed.toLong())
        Engine.getEngine("PyTorch").setRandomSeed(config.seed)

        TokenDataset(config.datasetDir).use { dataset ->
            val resumeDir = when (config.initFrom) {
                "resume" -> (config.resumeFrom ?: config.outDir).let(Checkpoints::resolveForResume)
                else -> null
            }
            val resumeMetadata = resumeDir?.let(Checkpoints::loadMetadata)
            val datasetVocabSize = dataset.inferredVocabSize()
                ?: values["vocab_size"]?.toInt()
                ?: resumeMetadata?.modelConfig?.vocabSize
                ?: error("Could not infer vocab size. Provide --vocab_size or include vocab.txt in dataset_dir.")
            val initialModelConfig = when {
                config.initFrom == "resume" -> resumeMetadata?.modelConfig
                    ?: error("No checkpoint metadata found to resume from.")
                config.initFrom.startsWith("gpt2") -> Gpt2Importer.modelConfig(config.initFrom, config.blockSize, config.dropout)
                else -> config.toModelConfig(datasetVocabSize)
            }
            require(initialModelConfig.vocabSize == datasetVocabSize || config.initFrom.startsWith("gpt2")) {
                "Model vocab_size=${initialModelConfig.vocabSize} does not match dataset vocab_size=$datasetVocabSize"
            }

            val lrSchedule = NanoGptLearningRate(config)
            val tokenLoss = SoftmaxCrossEntropyLoss("token_loss", 1f, 1, true, false)
            val optimizer = Optimizer.adamW()
                .optLearningRateTracker(lrSchedule)
                .optBeta1(config.beta1)
                .optBeta2(config.beta2)
                .optWeightDecays(0f)
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
                val gptModel = GptModel(initialModelConfig)
                model.block = gptModel
                if (resumeDir != null) {
                    model.load(resumeDir, CheckpointFiles.MODEL_PREFIX)
                }
                model.newTrainer(trainingConfig).use { trainer ->
                    trainer.initialize(Shape(config.batchSize.toLong(), config.blockSize.toLong()))
                    val modelParameters = model.block.parameters

                    when {
                        resumeDir != null -> {
                            val restored = OptimizerStateManager.load(resumeDir, optimizer, modelParameters, trainer.manager)
                            println("resumed_from=$resumeDir")
                            println(
                                if (restored) {
                                    "resume_note=optimizer and RNG state restored"
                                } else {
                                    "resume_note=optimizer state missing; falling back to weight-only resume"
                                },
                            )
                        }
                        config.initFrom.startsWith("gpt2") -> {
                            Gpt2Importer.initializePretrained(
                                model = model,
                                manager = trainer.manager,
                                modelType = config.initFrom,
                                dropout = config.dropout,
                            )
                            println("initialized_from=${config.initFrom}")
                        }
                        else -> Unit
                    }

                    val codec = dataset.checkpointCodec()
                    val parameterSummary = summarizeParameters(modelParameters)
                    println("parameters=${"%.2f".format(gptModel.parameterCount(nonEmbedding = true) / 1_000_000.0)}M")
                    println(
                        "num decayed parameter tensors: ${parameterSummary.decayedTensors}, " +
                            "with ${parameterSummary.decayedParams} parameters",
                    )
                    println(
                        "num non-decayed parameter tensors: ${parameterSummary.nonDecayedTensors}, " +
                            "with ${parameterSummary.nonDecayedParams} parameters",
                    )

                    val resumeState = resumeMetadata?.trainingState
                    var completedIters = resumeState?.iter ?: 0
                    var bestValLoss = resumeState?.bestValLoss ?: Float.POSITIVE_INFINITY
                    val trainingRandom = StatefulRandom(
                        if ((resumeState?.trainRngState ?: 0L) != 0L) resumeState!!.trainRngState else config.seed.toLong(),
                    )
                    val trainEvalRandom = StatefulRandom(
                        if ((resumeState?.trainEvalRngState ?: 0L) != 0L) resumeState!!.trainEvalRngState else (config.seed + 1).toLong(),
                    )
                    val valEvalRandom = StatefulRandom(
                        if ((resumeState?.valEvalRngState ?: 0L) != 0L) resumeState!!.valEvalRngState else (config.seed + 2).toLong(),
                    )
                    var lastLogTime = System.nanoTime()

                    saveLatest(model, optimizer, config.outDir, initialModelConfig, codec, completedIters, bestValLoss, trainingRandom, trainEvalRandom, valEvalRandom)

                    while (true) {
                        if (completedIters % config.evalInterval == 0) {
                            val trainLoss = estimateLoss(trainer, tokenLoss, dataset, Split.TRAIN, config, trainEvalRandom)
                            val valLoss = estimateLoss(trainer, tokenLoss, dataset, Split.VAL, config, valEvalRandom)
                            if (valLoss < bestValLoss) {
                                bestValLoss = valLoss
                            }

                            saveLatest(model, optimizer, config.outDir, initialModelConfig, codec, completedIters, bestValLoss, trainingRandom, trainEvalRandom, valEvalRandom)
                            if (valLoss <= bestValLoss || config.alwaysSaveCheckpoint) {
                                saveBest(model, optimizer, config.outDir, initialModelConfig, codec, completedIters, bestValLoss, trainingRandom, trainEvalRandom, valEvalRandom)
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

                        val currentLr = lrSchedule.getNewValue(completedIters)
                        val lossValue = trainIteration(
                            trainer = trainer,
                            tokenLoss = tokenLoss,
                            dataset = dataset,
                            config = config,
                            random = trainingRandom,
                            modelParameters = modelParameters,
                            currentLearningRate = currentLr,
                        )
                        completedIters += 1
                        if (completedIters % config.logInterval == 0) {
                            val now = System.nanoTime()
                            val millis = (now - lastLogTime) / 1_000_000.0
                            lastLogTime = now
                            println("iter $completedIters: loss ${"%.4f".format(lossValue)}, time ${"%.2f".format(millis)}ms")
                        }
                    }

                    saveLatest(model, optimizer, config.outDir, initialModelConfig, codec, completedIters, bestValLoss, trainingRandom, trainEvalRandom, valEvalRandom)
                }
            }
        }
    }

    private fun trainIteration(
        trainer: Trainer,
        tokenLoss: SoftmaxCrossEntropyLoss,
        dataset: TokenDataset,
        config: TrainConfig,
        random: StatefulRandom,
        modelParameters: ai.djl.nn.ParameterList,
        currentLearningRate: Float,
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
        applyDecoupledWeightDecay(modelParameters, currentLearningRate, config.weightDecay)
        trainer.step()
        return totalLoss / config.gradientAccumulationSteps
    }

    private fun estimateLoss(
        trainer: Trainer,
        tokenLoss: SoftmaxCrossEntropyLoss,
        dataset: TokenDataset,
        split: Split,
        config: TrainConfig,
        random: StatefulRandom,
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
        labels: NDArray,
        predictions: NDList,
    ): NDArray {
        val logProbs = predictions.singletonOrThrow()
        val vocabSize = logProbs.shape.get(logProbs.shape.dimension() - 1)
        val flatLogProbs = logProbs.reshape(-1, vocabSize)
        val flatLabels = labels.reshape(-1)
        return tokenLoss.evaluate(NDList(flatLabels), NDList(flatLogProbs))
    }

    private fun applyDecoupledWeightDecay(
        parameters: ai.djl.nn.ParameterList,
        learningRate: Float,
        weightDecay: Float,
    ) {
        if (weightDecay == 0f) {
            return
        }
        val scale = 1f - learningRate * weightDecay
        parameters.values()
            .mapNotNull { it.array }
            .filter { it.shape.dimension() >= 2 }
            .forEach { it.muli(scale) }
    }

    private fun saveLatest(
        model: Model,
        optimizer: Optimizer,
        outDir: java.nio.file.Path,
        modelConfig: GptConfig,
        codec: TextCodec,
        completedIters: Int,
        bestValLoss: Float,
        trainingRandom: StatefulRandom,
        trainEvalRandom: StatefulRandom,
        valEvalRandom: StatefulRandom,
    ) {
        saveCheckpoint(
            model = model,
            optimizer = optimizer,
            outputDir = Checkpoints.latestDir(outDir),
            modelConfig = modelConfig,
            codec = codec,
            completedIters = completedIters,
            bestValLoss = bestValLoss,
            trainingRandom = trainingRandom,
            trainEvalRandom = trainEvalRandom,
            valEvalRandom = valEvalRandom,
        )
    }

    private fun saveBest(
        model: Model,
        optimizer: Optimizer,
        outDir: java.nio.file.Path,
        modelConfig: GptConfig,
        codec: TextCodec,
        completedIters: Int,
        bestValLoss: Float,
        trainingRandom: StatefulRandom,
        trainEvalRandom: StatefulRandom,
        valEvalRandom: StatefulRandom,
    ) {
        saveCheckpoint(
            model = model,
            optimizer = optimizer,
            outputDir = Checkpoints.bestDir(outDir),
            modelConfig = modelConfig,
            codec = codec,
            completedIters = completedIters,
            bestValLoss = bestValLoss,
            trainingRandom = trainingRandom,
            trainEvalRandom = trainEvalRandom,
            valEvalRandom = valEvalRandom,
        )
    }

    private fun saveCheckpoint(
        model: Model,
        optimizer: Optimizer,
        outputDir: java.nio.file.Path,
        modelConfig: GptConfig,
        codec: TextCodec,
        completedIters: Int,
        bestValLoss: Float,
        trainingRandom: StatefulRandom,
        trainEvalRandom: StatefulRandom,
        valEvalRandom: StatefulRandom,
    ) {
        val trainingState = TrainingState(
            iter = completedIters,
            bestValLoss = bestValLoss,
            trainRngState = trainingRandom.currentState(),
            trainEvalRngState = trainEvalRandom.currentState(),
            valEvalRngState = valEvalRandom.currentState(),
        )
        Checkpoints.save(model, outputDir, modelConfig, codec, trainingState)
        OptimizerStateManager.save(outputDir, optimizer, model.block.parameters)
    }

    private fun summarizeParameters(parameters: ai.djl.nn.ParameterList): ParameterSummary {
        var decayedTensors = 0
        var decayedParams = 0L
        var nonDecayedTensors = 0
        var nonDecayedParams = 0L
        parameters.values()
            .mapNotNull { it.array }
            .forEach { array ->
                if (array.shape.dimension() >= 2) {
                    decayedTensors += 1
                    decayedParams += array.size()
                } else {
                    nonDecayedTensors += 1
                    nonDecayedParams += array.size()
                }
            }
        return ParameterSummary(decayedTensors, decayedParams, nonDecayedTensors, nonDecayedParams)
    }

    private data class ParameterSummary(
        val decayedTensors: Int,
        val decayedParams: Long,
        val nonDecayedTensors: Int,
        val nonDecayedParams: Long,
    )
}

private class NanoGptLearningRate(
    private val config: TrainConfig,
) : Tracker {
    override fun getNewValue(numUpdate: Int): Float {
        return if (!config.decayLr) {
            config.learningRate
        } else {
            when {
                numUpdate < config.warmupIters -> config.learningRate * (numUpdate + 1f) / (config.warmupIters + 1f)
                numUpdate > config.lrDecayIters -> config.minLr
                else -> {
                    val decayRatio = (numUpdate - config.warmupIters).toFloat() / (config.lrDecayIters - config.warmupIters).toFloat()
                    val coeff = 0.5f * (1.0f + kotlin.math.cos(Math.PI.toFloat() * decayRatio))
                    config.minLr + coeff * (config.learningRate - config.minLr)
                }
            }
        }
    }

    override fun getNewValue(parameterId: String, numUpdate: Int): Float = getNewValue(numUpdate)
}
