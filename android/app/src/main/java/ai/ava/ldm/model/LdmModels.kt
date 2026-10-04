package ai.ava.ldm.model

import org.json.JSONObject

/**
 * User profile configuration for personalization in AVA LDM.
 */
data class UserProfile(
    val userId: String = "default_user",
    val preferredLanguage: String = "Tanglish",
    val preferredDialect: String = "Standard",
    val responseStyle: String = "Concise"
)

/**
 * Breakdown of confidence scores across pipeline stages.
 */
data class ConfidenceScores(
    val language: Float = 0.95f,
    val dialect: Float = 0.85f,
    val intent: Float = 0.90f,
    val overall: Float = 0.90f
)

/**
 * Full linguistic understanding output produced by LdmProcessor.
 * The LDM produces this structured understanding without executing Android actions.
 */
data class LdmAnalysis(
    val transcript: String,
    val language: String,
    val dialect: String,
    val codeMixed: Boolean,
    val style: String,
    val normalizedText: String,
    val intent: String,
    val entities: Map<String, String> = emptyMap(),
    val confidence: ConfidenceScores = ConfidenceScores(),
    val processingTimeMs: Double = 0.0,
    val pipelineStep: String = "LLM Handoff Ready"
) {
    /**
     * Converts the analysis into the required standardized JSON format for LLM handoff.
     */
    fun toHandoffJson(): String {
        val root = JSONObject()
        root.put("normalized_text", normalizedText)
        root.put("language", language)
        root.put("dialect", dialect)
        root.put("code_mix", codeMixed)
        root.put("intent", intent)

        val entObj = JSONObject()
        entities.forEach { (k, v) -> entObj.put(k, v) }
        root.put("entities", entObj)

        val confObj = JSONObject()
        confObj.put("language", confidence.language.toDouble())
        confObj.put("dialect", confidence.dialect.toDouble())
        confObj.put("intent", confidence.intent.toDouble())
        confObj.put("overall", confidence.overall.toDouble())
        root.put("confidence", confObj)

        return root.toString(2)
    }
}

/**
 * An item in the synthetic prototype dataset.
 */
data class DatasetItem(
    val id: Int,
    val input: String,
    val language: String,
    val dialect: String,
    val codeMix: Boolean,
    val normalized: String,
    val intent: String
)

