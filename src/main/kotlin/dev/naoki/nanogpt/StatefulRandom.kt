package dev.naoki.nanogpt

class StatefulRandom(seed: Long) {
    private var state: Long = seed

    fun currentState(): Long = state

    fun nextLong(): Long {
        state += GOLDEN_GAMMA
        var z = state
        z = (z xor (z ushr 30)) * -4658895280553007687L
        z = (z xor (z ushr 27)) * -7723592293110705685L
        return z xor (z ushr 31)
    }

    fun nextLong(bound: Long): Long {
        require(bound > 0L) { "bound must be positive" }
        val mask = bound - 1
        if ((bound and mask) == 0L) {
            return nextLong() and mask
        }

        while (true) {
            val candidate = nextLong() ushr 1
            val result = candidate % bound
            if (candidate + mask - result >= 0L) {
                return result
            }
        }
    }

    companion object {
        private const val GOLDEN_GAMMA: Long = -7046029254386353131L
    }
}
