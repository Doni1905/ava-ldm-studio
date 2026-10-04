package ai.ava.ldm.ui

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import ai.ava.ldm.ui.theme.LdmStudioTheme

/**
 * Main Android Activity hosting the AVA Linguistic Dialect Model (LDM) Studio.
 *
 * Provides on-device linguistic analysis, normalization, intent parsing,
 * pipeline visualization, synthetic dataset exploration, and evaluation benchmarking.
 */
class LdmStudioActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            LdmStudioTheme {
                LdmStudioScreen()
            }
        }
    }
}

