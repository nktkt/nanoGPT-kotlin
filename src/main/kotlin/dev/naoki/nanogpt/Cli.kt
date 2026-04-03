package dev.naoki.nanogpt

import java.nio.file.Path
import kotlin.io.path.Path

object Cli {
    fun loadOverrides(args: Array<String>): Map<String, String> {
        val overrides = linkedMapOf<String, String>()
        args.filter { !it.startsWith("--") && it.endsWith(".properties") }
            .map(::Path)
            .forEach { path ->
                val props = PropertiesIO.load(path)
                props.stringPropertyNames().sorted().forEach { key ->
                    overrides[key] = props.getProperty(key)
                }
            }

        args.filter { it.startsWith("--") }.forEach { arg ->
            val stripped = arg.removePrefix("--")
            val key = stripped.substringBefore('=')
            val value = if ('=' in stripped) stripped.substringAfter('=') else "true"
            if (key == "config") {
                val props = PropertiesIO.load(Path(value))
                props.stringPropertyNames().sorted().forEach { propKey ->
                    overrides[propKey] = props.getProperty(propKey)
                }
            } else {
                overrides[key] = value
            }
        }
        return overrides
    }

    fun requirePath(values: Map<String, String>, key: String): Path {
        return values[key]?.let(::Path)
            ?: error("Missing required argument: --$key=/path")
    }

    fun string(values: Map<String, String>, key: String, default: String): String = values[key] ?: default
    fun int(values: Map<String, String>, key: String, default: Int): Int = values[key]?.toInt() ?: default
    fun float(values: Map<String, String>, key: String, default: Float): Float = values[key]?.toFloat() ?: default
    fun bool(values: Map<String, String>, key: String, default: Boolean): Boolean = values[key]?.toBooleanStrictOrNull() ?: default
}
