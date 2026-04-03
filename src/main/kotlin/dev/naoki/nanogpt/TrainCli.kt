package dev.naoki.nanogpt

import ai.djl.Model
import ai.djl.nn.Parameter
import ai.djl.training.DefaultTrainingConfig
import ai.djl.training.GradientCollector
import ai.djl.training.Trainer
import ai.djl.training.initializer.Initializer
import ai.djl.training.initializer.NormalInitializer
import ai.djl.training.loss.SoftmaxCrossEntropyLoss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import ai.djl.training.tracker.WarmUpTracker
import java.nio.file.Files
import java.util.Properties
import kotlin.io.path.Path
import kotlin.random.Random

object TrainCli {
    @JvmStatic
    fun main(args: Array<String>) {
        val values = Cli.loadOverrides(args)
        val config = TrainConfig(
            datasetDir = Cli.requirePath(values, "dataset_dir"),
            outDir = values["out_dir"]?.let(::Path) ?: Path("out"),
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

        Files.createDirectories(config.outDir)

        TokenDataset(config.datasetDir).use { dataset ->
            val vocabSize = dataset.inferredVocabSize()
                ?: values["vocab_size"]?.toInt()
                ?: error("Could not infer vocab size. Provide --vocab_size or include vocab.txt in dataset_dir.")
            val modelConfig = config.toModelConfig(vocabSize)
            val loss = SoftmaxCrossEntropyLoss("token_loss", 1f, -1, true, false)
            val tracker = buildTracker(config)
            val optimizer = Optimizer.adamW()
                .optLearningRateTracker(tracker)
                .optBeta1(config.beta1)
                .optBeta2(config.beta2)
                .optWeightDecays(config.weightDecay)
                .optClipGrad(config.gradClip)
                .build()

            val trainingConfig = DefaultTrainingConfig(loss)
                .optDevices(arrayOf(config.deviceHandle()))
                .optOptimizer(optimizer)
                .optInitializer(NormalInitializer(0.02f), Parameter.Type.WEIGHT)
                .optInitializer(Initializer.ONES, Parameter.Type.GAMMA)
                .optInitializer(Initializer.ZEROS, Parameter.Type.BETA)
                .optInitializer(Initializer.ZEROS, Parameter.Type.BIAS)

            Model.newInstance(CheckpointFiles.MODEL_PREFIX, "PyTorch").use { model ->
                model.block = GptModel(modelConfig)
                model.newTrainer(trainingConfig).use { trainer ->
                    trainer.initialize(
                        ai.djl.ndarray.types.Shape(config.batchSize.toLong(), config.blockSize.toLong()),
                    )
                    val numParams = countParameters(model)
                    println("parameters=${"%.2f".format(numParams / 1_000_000.0)}M")

                    val trainingRandom = Random(config.seed)
                    val validationRandom = Random(config.seed + 1)
                    var bestValLoss = Float.POSITIVE_INFINITY
                    var lastLogTime = System.nanoTime()

                    saveMetadata(config.outDir, modelConfig, dataset.checkpointCodec(), 0, bestValLoss)

                    for (iter in 0..config.maxIters) {
                        if (iter % config.evalInterval == 0) {
                            val trainLoss = estimateLoss(trainer, dataset, Split.TRAIN, config, validationRandom)
                            val valLoss = estimateLoss(trainer, dataset, Split.VAL, config, validationRandom)
                            println("step $iter: train loss ${"%.4f".format(trainLoss)}, val loss ${"%.4f".format(valLoss)}")
                            if (valLoss < bestValLoss || config.alwaysSaveCheckpoint) {
                                bestValLoss = valLoss
                                if (iter > 0) {
                                    saveCheckpoint(model, config.outDir, modelConfig, dataset.checkpointCodec(), iter, bestValLoss)
                                }
                            }
                            if (iter == 0 && config.evalOnly) {
                                break
                            }
                        }

                        val lossValue = trainIteration(trainer, dataset, config, trainingRandom)
                        if (iter % config.logInterval == 0) {
                            val now = System.nanoTime()
                            val millis = (now - lastLogTime) / 1_000_000.0
                            lastLogTime = now
                            println("iter $iter: loss ${"%.4f".format(lossValue)}, time ${"%.2f".format(millis)}ms")
                        }
                    }
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
                    val predictions = trainer.forward(ai.djl.ndarray.NDList(batch.data), ai.djl.ndarray.NDList(batch.labels))
                    val lossValue = trainer.loss.evaluate(ai.djl.ndarray.NDList(batch.labels), predictions)
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
                val predictions = trainer.evaluate(ai.djl.ndarray.NDList(batch.data))
                total += trainer.loss.evaluate(ai.djl.ndarray.NDList(batch.labels), predictions).getFloat()
            }
        }
        return total / config.evalIters
    }

    private fun saveCheckpoint(
        model: Model,
        outDir: java.nio.file.Path,
        modelConfig: GptConfig,
        codec: TextCodec,
        iter: Int,
        bestValLoss: Float,
    ) {
        model.save(outDir, CheckpointFiles.MODEL_PREFIX)
        saveMetadata(outDir, modelConfig, codec, iter, bestValLoss)
    }

    private fun saveMetadata(
        outDir: java.nio.file.Path,
        modelConfig: GptConfig,
        codec: TextCodec,
        iter: Int,
        bestValLoss: Float,
    ) {
        val modelProperties = Properties().apply {
            setProperty("block_size", modelConfig.blockSize.toString())
            setProperty("vocab_size", modelConfig.vocabSize.toString())
            setProperty("n_layer", modelConfig.nLayer.toString())
            setProperty("n_head", modelConfig.nHead.toString())
            setProperty("n_embd", modelConfig.nEmbd.toString())
            setProperty("dropout", modelConfig.dropout.toString())
            setProperty("bias", modelConfig.bias.toString())
            setProperty("codec", codec.type)
        }
        val stateProperties = Properties().apply {
            setProperty("iter", iter.toString())
            setProperty("best_val_loss", bestValLoss.toString())
        }
        PropertiesIO.store(outDir.resolve(CheckpointFiles.MODEL_PROPERTIES), modelProperties, "nanoGPT Kotlin model config")
        PropertiesIO.store(outDir.resolve(CheckpointFiles.STATE_PROPERTIES), stateProperties, "nanoGPT Kotlin training state")
        codec.copyArtifactsTo(outDir)
    }

    private fun countParameters(model: Model): Long {
        return model.block.parameters.values().sumOf { parameter ->
            parameter.array?.size() ?: 0L
        }
    }
}
