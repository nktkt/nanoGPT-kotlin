package dev.naoki.nanogpt

import ai.djl.Device
import ai.djl.ndarray.NDArray
import ai.djl.ndarray.NDManager
import ai.djl.nn.Parameter
import ai.djl.nn.ParameterList
import ai.djl.training.optimizer.Optimizer
import java.lang.reflect.Field
import java.nio.file.Files
import java.nio.file.Path
import java.util.Base64
import java.util.Properties
import java.util.concurrent.ConcurrentHashMap

object OptimizerStateManager {
    fun save(checkpointDir: Path, optimizer: Optimizer, parameters: ParameterList) {
        val stateFields = stateFields(optimizer) ?: return
        val optimizerDir = checkpointDir.resolve(CheckpointFiles.OPTIMIZER_DIR)
        Files.createDirectories(optimizerDir)

        val idToKey = parameters.associate { it.value.id to it.key }
        val manifest = Properties().apply {
            setProperty("optimizer_class", optimizer.javaClass.name)
        }

        val updateCounts = mutableMapOf<String, Int>()
        readField<Map<String, Int>>(optimizer, Optimizer::class.java, "updateCounts").forEach { (id, count) ->
            idToKey[id]?.let { stableKey ->
                updateCounts[stableKey] = count
                manifest.setProperty("update_count.$stableKey", count.toString())
            }
        }

        val means = readField<Map<String, Map<Device, NDArray>>>(optimizer, stateFields.ownerClass, "means")
        val variances = readField<Map<String, Map<Device, NDArray>>>(optimizer, stateFields.ownerClass, "variances")
        saveStateMap("mean", optimizerDir, manifest, idToKey, means)
        saveStateMap("variance", optimizerDir, manifest, idToKey, variances)

        val maxUpdate = updateCounts.values.maxOrNull() ?: 0
        manifest.setProperty("num_update", maxUpdate.toString())
        PropertiesIO.store(
            optimizerDir.resolve(CheckpointFiles.OPTIMIZER_MANIFEST),
            manifest,
            "nanoGPT Kotlin optimizer state",
        )
    }

    fun load(checkpointDir: Path, optimizer: Optimizer, parameters: ParameterList, manager: NDManager): Boolean {
        val stateFields = stateFields(optimizer) ?: return false
        val manifestPath = checkpointDir.resolve(CheckpointFiles.OPTIMIZER_DIR).resolve(CheckpointFiles.OPTIMIZER_MANIFEST)
        if (!Files.exists(manifestPath)) {
            return false
        }

        val manifest = PropertiesIO.load(manifestPath)
        val keyToParameter = parameters.associate { it.key to it.value }
        val updateCounts = ConcurrentHashMap<String, Int>()
        manifest.stringPropertyNames()
            .filter { it.startsWith("update_count.") }
            .forEach { key ->
                val stableKey = key.removePrefix("update_count.")
                val parameter = keyToParameter[stableKey] ?: return@forEach
                updateCounts[parameter.id] = manifest.getProperty(key).toInt()
            }

        val means = loadStateMap("mean", manifest, checkpointDir, keyToParameter, manager)
        val variances = loadStateMap("variance", manifest, checkpointDir, keyToParameter, manager)

        writeField(optimizer, Optimizer::class.java, "updateCounts", updateCounts)
        writeField(optimizer, Optimizer::class.java, "numUpdate", manifest.getProperty("num_update", "0").toInt())
        writeField(optimizer, stateFields.ownerClass, "means", means)
        writeField(optimizer, stateFields.ownerClass, "variances", variances)
        return true
    }

    private fun saveStateMap(
        kind: String,
        optimizerDir: Path,
        manifest: Properties,
        idToKey: Map<String, String>,
        stateMap: Map<String, Map<Device, NDArray>>,
    ) {
        stateMap.forEach { (id, deviceMap) ->
            val stableKey = idToKey[id] ?: return@forEach
            val stateArray = deviceMap.values.firstOrNull() ?: return@forEach
            val fileName = "$kind-${encodeKey(stableKey)}.nd"
            Files.write(optimizerDir.resolve(fileName), stateArray.toDevice(Device.cpu(), true).encode())
            manifest.setProperty("$kind.$stableKey", fileName)
        }
    }

    private fun loadStateMap(
        kind: String,
        manifest: Properties,
        checkpointDir: Path,
        keyToParameter: Map<String, Parameter>,
        manager: NDManager,
    ): MutableMap<String, MutableMap<Device, NDArray>> {
        val optimizerDir = checkpointDir.resolve(CheckpointFiles.OPTIMIZER_DIR)
        val result = ConcurrentHashMap<String, MutableMap<Device, NDArray>>()
        manifest.stringPropertyNames()
            .filter { it.startsWith("$kind.") }
            .forEach { key ->
                val stableKey = key.removePrefix("$kind.")
                val parameter = keyToParameter[stableKey] ?: return@forEach
                val fileName = manifest.getProperty(key)
                val bytes = Files.readAllBytes(optimizerDir.resolve(fileName))
                val decoded = manager.decode(bytes)
                val device = parameter.array.device
                val state = decoded.toDevice(device, true)
                result.computeIfAbsent(parameter.id) { ConcurrentHashMap() }[device] = state
            }
        return result
    }

    private fun stateFields(optimizer: Optimizer): StateFields? {
        val clazz = optimizer.javaClass
        return when {
            hasField(clazz, "means") && hasField(clazz, "variances") -> StateFields(clazz)
            else -> null
        }
    }

    private fun hasField(clazz: Class<*>, fieldName: String): Boolean {
        return runCatching { clazz.getDeclaredField(fieldName) }.isSuccess
    }

    private fun encodeKey(key: String): String {
        return Base64.getUrlEncoder().withoutPadding().encodeToString(key.toByteArray(Charsets.UTF_8))
    }

    private fun field(owner: Class<*>, name: String): Field {
        return owner.getDeclaredField(name).apply { isAccessible = true }
    }

    @Suppress("UNCHECKED_CAST")
    private fun <T> readField(target: Any, owner: Class<*>, name: String): T {
        return field(owner, name).get(target) as T
    }

    private fun writeField(target: Any, owner: Class<*>, name: String, value: Any) {
        field(owner, name).set(target, value)
    }

    private data class StateFields(
        val ownerClass: Class<*>,
    )
}
