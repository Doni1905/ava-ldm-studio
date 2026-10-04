package ai.ava.ldm

import ai.ava.ldm.core.RuleBasedLdmProcessor
import ai.ava.ldm.data.EvaluationEngine
import ai.ava.ldm.data.SyntheticDataset
import ai.ava.ldm.model.UserProfile
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test

class LdmProcessorTest {

    private lateinit var processor: RuleBasedLdmProcessor

    @Before
    fun setUp() {
        processor = RuleBasedLdmProcessor()
    }

    @Test
    fun testChennaiSlangNormalizationAndIntent() {
        val input = "Dei nalaiku assignment submit panna remind pannu"
        val analysis = processor.analyzeUtterance(input)

        assertEquals("Chennai", analysis.dialect)
        assertEquals("Tanglish", analysis.language)
        assertTrue(analysis.codeMixed)
        assertEquals("CREATE_REMINDER", analysis.intent)
        assertTrue(analysis.normalizedText.contains("Remind", ignoreCase = true))
        assertTrue(analysis.normalizedText.contains("assignment", ignoreCase = true))
        assertTrue(analysis.processingTimeMs >= 0.0)
    }

    @Test
    fun testMaduraiDialect() {
        val input = "Thambi ku oru message anuppu naan late ah varen nu"
        val analysis = processor.analyzeUtterance(input)

        assertEquals("Madurai", analysis.dialect)
        assertEquals("SEND_MESSAGE", analysis.intent)
        assertTrue(analysis.codeMixed)
    }

    @Test
    fun testKonguDialect() {
        val input = "Ayya nalaiku medicine saapida remind pannunga"
        val analysis = processor.analyzeUtterance(input)

        assertEquals("Kongu", analysis.dialect)
        assertEquals("CREATE_REMINDER", analysis.intent)
    }

    @Test
    fun testNellaiDialect() {
        val input = "Friend ku call pottu kudu pa"
        val analysis = processor.analyzeUtterance(input)

        assertEquals("Nellai", analysis.dialect)
        assertEquals("MAKE_CALL", analysis.intent)
    }

    @Test
    fun testStandardTamilAlarm() {
        val input = "Naalaikku kaalaila 6 maniku alarm vai"
        val analysis = processor.analyzeUtterance(input)

        assertEquals("SET_ALARM", analysis.intent)
        assertTrue(analysis.normalizedText.contains("alarm", ignoreCase = true))
    }

    @Test
    fun testEnglishQuery() {
        val input = "Set an alarm for 5:30 am"
        val analysis = processor.analyzeUtterance(input)

        assertEquals("English", analysis.language)
        assertEquals("Standard", analysis.dialect)
        assertFalse(analysis.codeMixed)
        assertEquals("SET_ALARM", analysis.intent)
    }

    @Test
    fun testHandoffJsonFormatting() {
        val input = "Machi inniku evening gym poga remind pannu"
        val analysis = processor.analyzeUtterance(input)
        val jsonString = analysis.toHandoffJson()

        assertNotNull(jsonString)
        val json = JSONObject(jsonString)

        assertTrue(json.has("normalized_text"))
        assertTrue(json.has("language"))
        assertTrue(json.has("dialect"))
        assertTrue(json.has("code_mix"))
        assertTrue(json.has("intent"))
        assertTrue(json.has("entities"))
        assertTrue(json.has("confidence"))

        assertEquals("CREATE_REMINDER", json.getString("intent"))
        assertEquals("Chennai", json.getString("dialect"))
        assertEquals(true, json.getBoolean("code_mix"))
    }

    @Test
    fun testPersonalizationProfileInfluence() {
        val input = "Meeting iruku remind pannu"
        val userProfile = UserProfile(preferredDialect = "Madurai")
        val analysis = processor.analyzeUtterance(input, userProfile)

        assertEquals("Madurai", analysis.dialect)
    }

    @Test
    fun testEvaluationEngineRunsSuccessfully() {
        val metrics = EvaluationEngine.evaluate(processor, SyntheticDataset.ITEMS)

        assertEquals(30, metrics.totalSamples)
        assertTrue("Expected intent accuracy > 75%, got ${metrics.intentAccuracy}", metrics.intentAccuracy >= 75.0)
        assertTrue("Expected dialect accuracy > 75%, got ${metrics.dialectAccuracy}", metrics.dialectAccuracy >= 75.0)
        assertTrue("Expected avg latency < 50ms, got ${metrics.avgLatencyMs}", metrics.avgLatencyMs < 50.0)
        assertEquals(30, metrics.details.size)
    }
}

