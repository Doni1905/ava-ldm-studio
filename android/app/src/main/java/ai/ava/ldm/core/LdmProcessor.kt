package ai.ava.ldm.core

import ai.ava.ldm.model.LdmAnalysis
import ai.ava.ldm.model.UserProfile

/**
 * Replaceable Kotlin interface for the Linguistic Dialect Model in AVA.
 *
 * Responsibilities:
 * - Understands Tamil, Tanglish, regional dialect, slang, and code-mixing.
 * - Normalizes user's intended meaning into structured semantic English for the local LLM.
 * - MUST NOT execute Android actions directly.
 * - MUST NOT behave as a general chatbot.
 */
interface LdmProcessor {
    /**
     * Analyzes raw speech transcript/text and user profile context,
     * returning structured linguistic understanding for LLM handoff.
     */
    fun analyzeUtterance(input: String, userProfile: UserProfile = UserProfile()): LdmAnalysis
}

