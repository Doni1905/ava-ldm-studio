package ai.ava.ldm.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowForward
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.ava.ldm.model.LdmAnalysis
import ai.ava.ldm.ui.theme.BgCard
import ai.ava.ldm.ui.theme.BgCardElevated
import ai.ava.ldm.ui.theme.BorderSubtle
import ai.ava.ldm.ui.theme.CyanAccent
import ai.ava.ldm.ui.theme.MintSuccess
import ai.ava.ldm.ui.theme.SkyBlue
import ai.ava.ldm.ui.theme.TextMuted
import ai.ava.ldm.ui.theme.TextPrimary
import ai.ava.ldm.ui.theme.TextSecondary

private data class PipelineStep(
    val stepNumber: Int,
    val title: String,
    val subtitle: String,
    val tagValue: String,
    val tagColor: Color
)

@Composable
fun PipelineSection(
    analysis: LdmAnalysis,
    modifier: Modifier = Modifier
) {
    val steps = listOf(
        PipelineStep(
            1,
            "ASR Transcript",
            "Audio → Text",
            if (analysis.transcript.isNotEmpty()) "Ready" else "Idle",
            SkyBlue
        ),
        PipelineStep(
            2,
            "Language",
            "Script & Token ID",
            analysis.language,
            CyanAccent
        ),
        PipelineStep(
            3,
            "Dialect",
            "Regional Classifier",
            analysis.dialect,
            SkyBlue
        ),
        PipelineStep(
            4,
            "Code-Mix",
            "CMI Index Proxy",
            if (analysis.codeMixed) "Mixed" else "Monolingual",
            CyanAccent
        ),
        PipelineStep(
            5,
            "Normalization",
            "Slang / Verbs Mapped",
            "Normalized",
            MintSuccess
        ),
        PipelineStep(
            6,
            "Intent & Entity",
            "Semantic Slots",
            analysis.intent,
            SkyBlue
        ),
        PipelineStep(
            7,
            "LLM Handoff",
            "Validated JSON",
            "Handoff",
            MintSuccess
        )
    )

    Column(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(BgCard)
            .border(1.dp, BorderSubtle, RoundedCornerShape(16.dp))
            .padding(16.dp)
    ) {
        // Section Header
        Text(
            text = "Pipeline Flow Architecture",
            color = TextPrimary,
            fontSize = 17.sp,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = "Linear linguistic understanding stages before local LLM handoff",
            color = TextSecondary,
            fontSize = 12.sp
        )

        Spacer(modifier = Modifier.height(14.dp))

        // Horizontal Step Sequence
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState()),
            verticalAlignment = Alignment.CenterVertically
        ) {
            steps.forEachIndexed { index, step ->
                PipelineNode(step = step)

                if (index < steps.size - 1) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.ArrowForward,
                        contentDescription = "to",
                        tint = BorderSubtle,
                        modifier = Modifier
                            .padding(horizontal = 6.dp)
                            .height(16.dp)
                            .width(16.dp)
                    )
                }
            }
        }
    }
}

@Composable
private fun PipelineNode(step: PipelineStep) {
    Column(
        modifier = Modifier
            .width(140.dp)
            .clip(RoundedCornerShape(10.dp))
            .background(BgCardElevated)
            .border(1.dp, BorderSubtle, RoundedCornerShape(10.dp))
            .padding(10.dp),
        horizontalAlignment = Alignment.Start
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
            modifier = Modifier.fillMaxWidth()
        ) {
            Box(
                modifier = Modifier
                    .clip(CircleShape)
                    .background(CyanAccent.copy(alpha = 0.2f))
                    .padding(horizontal = 6.dp, vertical = 2.dp),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = "${step.stepNumber}",
                    color = CyanAccent,
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Bold
                )
            }

            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(4.dp))
                    .background(step.tagColor.copy(alpha = 0.15f))
                    .padding(horizontal = 6.dp, vertical = 2.dp)
            ) {
                Text(
                    text = step.tagValue,
                    color = step.tagColor,
                    fontSize = 9.sp,
                    fontWeight = FontWeight.Bold,
                    maxLines = 1
                )
            }
        }

        Spacer(modifier = Modifier.height(8.dp))

        Text(
            text = step.title,
            color = TextPrimary,
            fontSize = 12.sp,
            fontWeight = FontWeight.SemiBold,
            maxLines = 1
        )
        Text(
            text = step.subtitle,
            color = TextMuted,
            fontSize = 10.sp,
            maxLines = 1
        )
    }
}

