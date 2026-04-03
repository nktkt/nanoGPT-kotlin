package dev.naoki.nanogpt

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDList
import ai.djl.ndarray.NDManager
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.nn.AbstractBlock
import ai.djl.nn.Activation
import ai.djl.nn.Parameter
import ai.djl.nn.core.Linear
import ai.djl.nn.norm.Dropout
import ai.djl.nn.norm.LayerNorm
import ai.djl.nn.transformer.IdEmbedding
import ai.djl.training.ParameterStore
import ai.djl.util.PairList
import kotlin.math.sqrt

class CausalSelfAttentionBlock(private val config: GptConfig) : AbstractBlock(VERSION) {
    private val cAttn = addChildBlock(
        "c_attn",
        Linear.builder().setUnits((3 * config.nEmbd).toLong()).optBias(config.bias).build(),
    )
    private val cProj = addChildBlock(
        "c_proj",
        Linear.builder().setUnits(config.nEmbd.toLong()).optBias(config.bias).build(),
    )
    private val attnDropout = addChildBlock("attn_dropout", Dropout.builder().optRate(config.dropout).build())
    private val residDropout = addChildBlock("resid_dropout", Dropout.builder().optRate(config.dropout).build())

    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> = arrayOf(inputShapes[0])

    override fun initializeChildBlocks(manager: NDManager, dataType: DataType, vararg inputShapes: Shape) {
        val projectionShape = Shape(-1, config.nEmbd.toLong())
        cAttn.initialize(manager, DataType.FLOAT32, projectionShape)
        cProj.initialize(manager, DataType.FLOAT32, projectionShape)
        attnDropout.initialize(manager, DataType.FLOAT32, inputShapes[0])
        residDropout.initialize(manager, DataType.FLOAT32, inputShapes[0])
    }

    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?,
    ): NDList {
        val x = inputs.singletonOrThrow()
        val batch = x.shape.get(0)
        val time = x.shape.get(1)
        val channels = x.shape.get(2)
        val headSize = config.nEmbd / config.nHead

        val qkv = cAttn.forward(parameterStore, NDList(x), training, params).singletonOrThrow()
        val parts = qkv.split(3, 2)
        val q = createHeads(parts[0], batch, time, headSize)
        val k = createHeads(parts[1], batch, time, headSize)
        val v = createHeads(parts[2], batch, time, headSize)

        val scores = q.matMul(k.transpose(0, 1, 3, 2)).mul(1f / sqrt(headSize.toFloat()))
        val maskOffset = causalMask(x.manager, time)
            .logicalNot()
            .toType(DataType.FLOAT32, false)
            .mul(-1.0e9f)
            .reshape(1, 1, time, time)
        val probs = attnDropout
            .forward(parameterStore, NDList(scores.add(maskOffset).softmax(3)), training, params)
            .singletonOrThrow()
        val attended = probs.matMul(v)
            .transpose(0, 2, 1, 3)
            .reshape(batch, time, channels)
        val projected = cProj.forward(parameterStore, NDList(attended), training, params).singletonOrThrow()
        return residDropout.forward(parameterStore, NDList(projected), training, params)
    }

    fun qkvWeight(): Parameter = cAttn.parameters.get("weight")
    fun qkvBias(): Parameter? = if (cAttn.parameters.contains("bias")) cAttn.parameters.get("bias") else null
    fun projWeight(): Parameter = cProj.parameters.get("weight")
    fun projBias(): Parameter? = if (cProj.parameters.contains("bias")) cProj.parameters.get("bias") else null

    private fun createHeads(array: NDArray, batch: Long, time: Long, headSize: Int): NDArray {
        return array.reshape(batch, time, config.nHead.toLong(), headSize.toLong()).transpose(0, 2, 1, 3)
    }

    private fun causalMask(manager: NDManager, time: Long): NDArray {
        val positions = manager.arange(time.toInt())
        return positions.reshape(Shape(time, 1)).gte(positions.reshape(Shape(1, time)))
    }

    companion object {
        private const val VERSION: Byte = 1
    }
}

class MlpBlock(private val config: GptConfig) : AbstractBlock(VERSION) {
    private val cFc = addChildBlock(
        "c_fc",
        Linear.builder().setUnits((4 * config.nEmbd).toLong()).optBias(config.bias).build(),
    )
    private val gelu = addChildBlock("gelu", Activation.geluBlock())
    private val cProj = addChildBlock(
        "c_proj",
        Linear.builder().setUnits(config.nEmbd.toLong()).optBias(config.bias).build(),
    )
    private val dropout = addChildBlock("dropout", Dropout.builder().optRate(config.dropout).build())

    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> = arrayOf(inputShapes[0])

    override fun initializeChildBlocks(manager: NDManager, dataType: DataType, vararg inputShapes: Shape) {
        cFc.initialize(manager, DataType.FLOAT32, inputShapes[0])
        val fcOutputShape = cFc.getOutputShapes(inputShapes)[0]
        gelu.initialize(manager, DataType.FLOAT32, fcOutputShape)
        cProj.initialize(manager, DataType.FLOAT32, fcOutputShape)
        val projOutputShape = cProj.getOutputShapes(arrayOf(fcOutputShape))[0]
        dropout.initialize(manager, DataType.FLOAT32, projOutputShape)
    }

    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?,
    ): NDList {
        var x = cFc.forward(parameterStore, inputs, training, params)
        x = gelu.forward(parameterStore, x, training, params)
        x = cProj.forward(parameterStore, x, training, params)
        return dropout.forward(parameterStore, x, training, params)
    }

    fun fcWeight(): Parameter = cFc.parameters.get("weight")
    fun fcBias(): Parameter? = if (cFc.parameters.contains("bias")) cFc.parameters.get("bias") else null
    fun projWeight(): Parameter = cProj.parameters.get("weight")
    fun projBias(): Parameter? = if (cProj.parameters.contains("bias")) cProj.parameters.get("bias") else null

    companion object {
        private const val VERSION: Byte = 1
    }
}

class TransformerBlock(private val config: GptConfig) : AbstractBlock(VERSION) {
    private val ln1 = addChildBlock(
        "ln_1",
        LayerNorm.builder().axis(2).optCenter(config.bias).optScale(true).build(),
    )
    private val attn = addChildBlock("attn", CausalSelfAttentionBlock(config))
    private val ln2 = addChildBlock(
        "ln_2",
        LayerNorm.builder().axis(2).optCenter(config.bias).optScale(true).build(),
    )
    private val mlp = addChildBlock("mlp", MlpBlock(config))

    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> = arrayOf(inputShapes[0])

    override fun initializeChildBlocks(manager: NDManager, dataType: DataType, vararg inputShapes: Shape) {
        val hiddenShape = inputShapes[0]
        ln1.initialize(manager, DataType.FLOAT32, hiddenShape)
        attn.initialize(manager, DataType.FLOAT32, hiddenShape)
        ln2.initialize(manager, DataType.FLOAT32, hiddenShape)
        mlp.initialize(manager, DataType.FLOAT32, hiddenShape)
    }

    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?,
    ): NDList {
        var x = inputs.singletonOrThrow()
        val attnOut = attn.forward(parameterStore, ln1.forward(parameterStore, NDList(x), training, params), training, params)
            .singletonOrThrow()
        x = x.add(attnOut)
        val mlpOut = mlp.forward(parameterStore, ln2.forward(parameterStore, NDList(x), training, params), training, params)
            .singletonOrThrow()
        return NDList(x.add(mlpOut))
    }

    fun ln1Weight(): Parameter = ln1.parameters.get("gamma")
    fun ln1Bias(): Parameter? = if (ln1.parameters.contains("beta")) ln1.parameters.get("beta") else null
    fun ln2Weight(): Parameter = ln2.parameters.get("gamma")
    fun ln2Bias(): Parameter? = if (ln2.parameters.contains("beta")) ln2.parameters.get("beta") else null
    fun attention(): CausalSelfAttentionBlock = attn
    fun mlpBlock(): MlpBlock = mlp

    companion object {
        private const val VERSION: Byte = 1
    }
}

class GptModel(private val config: GptConfig) : AbstractBlock(VERSION) {
    private val tokenEmbedding = addChildBlock(
        "wte",
        IdEmbedding.Builder().setDictionarySize(config.vocabSize).setEmbeddingSize(config.nEmbd).build(),
    )
    private val positionEmbedding = addChildBlock(
        "wpe",
        IdEmbedding.Builder().setDictionarySize(config.blockSize).setEmbeddingSize(config.nEmbd).build(),
    )
    private val dropout = addChildBlock("drop", Dropout.builder().optRate(config.dropout).build())
    private val blocks: List<TransformerBlock> = List(config.nLayer) { index ->
        addChildBlock("h$index", TransformerBlock(config))
    }
    private val finalNorm = addChildBlock(
        "ln_f",
        LayerNorm.builder().axis(2).optCenter(config.bias).optScale(true).build(),
    )

    override fun getOutputShapes(inputShapes: Array<Shape>): Array<Shape> {
        val input = inputShapes[0]
        return arrayOf(Shape(input.get(0), input.get(1), config.vocabSize.toLong()))
    }

    override fun initializeChildBlocks(manager: NDManager, dataType: DataType, vararg inputShapes: Shape) {
        val tokenIds = inputShapes[0]
        val hiddenShape = Shape(tokenIds.get(0), tokenIds.get(1), config.nEmbd.toLong())
        tokenEmbedding.initialize(manager, DataType.FLOAT32, tokenIds)
        positionEmbedding.initialize(manager, DataType.FLOAT32, Shape(tokenIds.get(1)))
        dropout.initialize(manager, DataType.FLOAT32, hiddenShape)
        blocks.forEach { block -> block.initialize(manager, DataType.FLOAT32, hiddenShape) }
        finalNorm.initialize(manager, DataType.FLOAT32, hiddenShape)
    }

    override fun forwardInternal(
        parameterStore: ParameterStore,
        inputs: NDList,
        training: Boolean,
        params: PairList<String, Any>?,
    ): NDList {
        val idx = inputs.singletonOrThrow()
        val time = idx.shape.get(1)
        require(time <= config.blockSize.toLong()) {
            "Cannot forward sequence length $time with block_size=${config.blockSize}"
        }

        val positions = idx.manager.arange(time.toInt()).reshape(Shape(1, time))
        val tokEmb = tokenEmbedding.forward(parameterStore, NDList(idx), training, params).singletonOrThrow()
        val posEmb = positionEmbedding.forward(parameterStore, NDList(positions), training, params).singletonOrThrow()
        var x = dropout.forward(parameterStore, NDList(tokEmb.add(posEmb)), training, params).singletonOrThrow()
        blocks.forEach { block ->
            x = block.forward(parameterStore, NDList(x), training, params).singletonOrThrow()
        }
        x = finalNorm.forward(parameterStore, NDList(x), training, params).singletonOrThrow()
        return NDList(tokenEmbedding.probabilities(parameterStore, x, training))
    }

    fun tokenEmbeddingWeight(): Parameter = tokenEmbedding.parameters.get("embedding")
    fun positionEmbeddingWeight(): Parameter = positionEmbedding.parameters.get("embedding")
    fun finalNormWeight(): Parameter = finalNorm.parameters.get("gamma")
    fun finalNormBias(): Parameter? = if (finalNorm.parameters.contains("beta")) finalNorm.parameters.get("beta") else null
    fun transformerBlock(index: Int): TransformerBlock = blocks[index]
    fun config(): GptConfig = config
    fun parameterCount(nonEmbedding: Boolean = true): Long {
        var total = getParameters().values().sumOf { it.array?.size() ?: 0L }
        if (nonEmbedding) {
            total -= positionEmbeddingWeight().array.size()
        }
        return total
    }

    companion object {
        private const val VERSION: Byte = 1
    }
}
