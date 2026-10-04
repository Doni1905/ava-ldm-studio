package ai.ava.ldm.data

import ai.ava.ldm.core.LdmProcessor
import ai.ava.ldm.model.DatasetItem
import ai.ava.ldm.model.LdmAnalysis

/**
 * Result metrics from running an evaluation over dataset items.
 */
data class EvaluationMetrics(
    val totalSamples: Int,
    val normalizationAccuracy: Double,
    val intentAccuracy: Double,
    val dialectAccuracy: Double,
    val codeMixAccuracy: Double,
    val avgLatencyMs: Double,
    val details: List<EvaluationDetailItem> = emptyList()
)

/**
 * Fine-grained per-utterance evaluation detail.
 */
data class EvaluationDetailItem(
    val id: Int,
    val input: String,
    val expectedIntent: String,
    val predictedIntent: String,
    val intentMatched: Boolean,
    val expectedDialect: String,
    val predictedDialect: String,
    val dialectMatched: Boolean,
    val expectedNormalized: String,
    val predictedNormalized: String,
    val normalizedMatched: Boolean,
    val expectedCodeMix: Boolean,
    val predictedCodeMix: Boolean,
    val codeMixMatched: Boolean,
    val latencyMs: Double
)

/**
 * Engine for benchmarking the LDM processor on prototype dataset utterances.
 */
object EvaluationEngine {

    /**
     * Executes evaluation of the given [processor] on the provided [dataset].
     * Defaults to [SyntheticDataset.ITEMS].
     */
    fun evaluate(
        processor: LdmProcessor,
        dataset: List<DatasetItem> = SyntheticDataset.ITEMS
    ): EvaluationMetrics {
        if (dataset.isEmpty()) {
            return EvaluationMetrics(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        }

        var intentCorrect = 0
        var dialectCorrect = 0
        var normCorrect = 0
        var codeMixCorrect = 0
        var totalLatency = 0.0

        val details = ArrayList<EvaluationDetailItem>(dataset.size)

        for (item in dataset) {
            val analysis: LdmAnalysis = processor.analyzeUtterance(item.input)
            totalLatency += analysis.processingTimeMs

            val intentMatch = analysis.intent.equals(item.intent, ignoreCase = true)
            if (intentMatch) intentCorrect++

            val dialectMatch = analysis.dialect.equals(item.dialect, ignoreCase = true)
            if (dialectMatch) dialectCorrect++

            val codeMixMatch = analysis.codeMixed == item.codeMix
            if (codeMixMatch) codeMixCorrect++

            // Normalization is matched either by exact match or normalized semantic token overlap
            val normMatch = isSemanticMatch(analysis.normalizedText, item.normalized)
            if (normMatch) normCorrect++

            details.add(
                EvaluationDetailItem(
                    id = item.id,
                    input = item.input,
                    expectedIntent = item.intent,
                    predictedIntent = analysis.intent,
                    intentMatched = intentMatch,
                    expectedDialect = item.dialect,
                    predictedDialect = analysis.dialect,
                    dialectMatched = dialectMatch,
                    expectedNormalized = item.normalized,
                    predictedNormalized = analysis.normalizedText,
                    normalizedMatched = normMatch,
                    expectedCodeMix = item.codeMix,
                    predictedCodeMix = analysis.codeMixed,
                    codeMixMatched = codeMixMatch,
                    latencyMs = analysis.processingTimeMs
                )
            )
        }

        val count = dataset.size.toDouble()
        return EvaluationMetrics(
            totalSamples = dataset.size,
            normalizationAccuracy = (normCorrect / count) * 100.0,
            intentAccuracy = (intentCorrect / count) * 100.0,
            dialectAccuracy = (dialectCorrect / count) * 100.0,
            codeMixAccuracy = (codeMixCorrect / count) * 100.0,
            avgLatencyMs = totalLatency / count,
            details = details
        )
    }

    private fun isSemanticMatch(predicted: String, expected: String): Boolean {
        return clean(predicted) == clean(expected)
    }

    private fun clean(text: String): String {
        return text.lowercase()
            .replace(Regex("[^\\p{L}\\p{N}\\s]"), "")
            .replace(Regex("\\s+"), " ")
            .trim()
    }
}

