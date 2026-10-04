package ai.ava.ldm.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AutoAwesome
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material3.Icon
import androidx.compose.material3.LinearProgressIndicator
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
import ai.ava.ldm.ui.theme.AmberWarning
import ai.ava.ldm.ui.theme.BgCard
import ai.ava.ldm.ui.theme.BgCardElevated
import ai.ava.ldm.ui.theme.BorderSubtle
import ai.ava.ldm.ui.theme.CyanAccent
import ai.ava.ldm.ui.theme.MintSuccess
import ai.ava.ldm.ui.theme.PurpleBadge
import ai.ava.ldm.ui.theme.SkyBlue
import ai.ava.ldm.ui.theme.TextMuted
import ai.ava.ldm.ui.theme.TextPrimary
import ai.ava.ldm.ui.theme.TextSecondary
import java.util.Locale

@OptIn(ExperimentalLayoutApi::class)
@Composable
fun AnalysisSection(
    analysis: LdmAnalysis,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(BgCard)
            .border(1.dp, BorderSubtle, RoundedCornerShape(16.dp))
            .padding(16.dp)
    ) {
        // Section Header with Execution Latency
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text(
                    text = "Linguistic Analysis Breakdown",
                    color = TextPrimary,
                    fontSize = 17.sp,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "Zero-cloud, on-device dialect understanding & normalization",
                    color = TextSecondary,
                    fontSize = 12.sp
                )
            }

            // Latency Badge
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(8.dp))
                    .background(MintSuccess.copy(alpha = 0.15f))
                    .border(1.dp, MintSuccess.copy(alpha = 0.3f), RoundedCornerShape(8.dp))
                    .padding(horizontal = 10.dp, vertical = 5.dp)
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.Bolt,
                        contentDescription = null,
                        tint = MintSuccess,
                        modifier = Modifier.height(14.dp).width(14.dp)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = String.format(Locale.US, "%.2f ms", analysis.processingTimeMs),
                        color = MintSuccess,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(14.dp))

        // Attribute Badges FlowRow
        FlowRow(
            horizontalArrangement = Arrangement.spacedBy(8.dp),
            verticalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            LdmBadge(label = "Language", value = analysis.language, color = CyanAccent)
            LdmBadge(label = "Dialect", value = analysis.dialect, color = SkyBlue)
            LdmBadge(
                label = "Code-Mixed",
                value = if (analysis.codeMixed) "Yes" else "No",
                color = if (analysis.codeMixed) AmberWarning else TextSecondary
            )
            LdmBadge(label = "Style", value = analysis.style, color = PurpleBadge)
        }

        Spacer(modifier = Modifier.height(14.dp))

        // Normalized Semantic Output Card
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(12.dp))
                .background(BgCardElevated)
                .border(1.dp, CyanAccent.copy(alpha = 0.3f), RoundedCornerShape(12.dp))
                .padding(14.dp)
        ) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Icon(
                    imageVector = Icons.Default.AutoAwesome,
                    contentDescription = null,
                    tint = CyanAccent,
                    modifier = Modifier.height(16.dp).width(16.dp)
                )
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    text = "NORMALIZED SEMANTIC ENGLISH",
                    color = CyanAccent,
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 0.8.sp
                )
            }

            Spacer(modifier = Modifier.height(6.dp))

            Text(
                text = if (analysis.normalizedText.isNotEmpty()) {
                    analysis.normalizedText
                } else {
                    "No text analyzed yet."
                },
                color = TextPrimary,
                fontSize = 15.sp,
                fontWeight = FontWeight.SemiBold,
                lineHeight = 22.sp
            )
        }

        Spacer(modifier = Modifier.height(14.dp))

        // Intent & Extracted Entities
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            // Intent Box
            Column(
                modifier = Modifier
                    .weight(1f)
                    .clip(RoundedCornerShape(10.dp))
                    .background(BgCardElevated)
                    .border(1.dp, BorderSubtle, RoundedCornerShape(10.dp))
                    .padding(12.dp)
            ) {
                Text(
                    text = "INTENT",
                    color = TextMuted,
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(4.dp))
                Text(
                    text = analysis.intent,
                    color = PurpleBadge,
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Bold
                )
            }

            // Confidence Box
            Column(
                modifier = Modifier
                    .weight(1f)
                    .clip(RoundedCornerShape(10.dp))
                    .background(BgCardElevated)
                    .border(1.dp, BorderSubtle, RoundedCornerShape(10.dp))
                    .padding(12.dp)
            ) {
                Text(
                    text = "OVERALL CONFIDENCE",
                    color = TextMuted,
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(4.dp))
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.CheckCircle,
                        contentDescription = null,
                        tint = MintSuccess,
                        modifier = Modifier.height(14.dp).width(14.dp)
                    )
                    Spacer(modifier = Modifier.width(4.dp))
                    Text(
                        text = "${(analysis.confidence.overall * 100).toInt()}%",
                        color = MintSuccess,
                        fontSize = 13.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }

        // Entities List
        if (analysis.entities.isNotEmpty()) {
            Spacer(modifier = Modifier.height(10.dp))
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(10.dp))
                    .background(BgCardElevated)
                    .border(1.dp, BorderSubtle, RoundedCornerShape(10.dp))
                    .padding(12.dp)
            ) {
                Text(
                    text = "EXTRACTED ENTITIES",
                    color = TextMuted,
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(6.dp))
                FlowRow(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    verticalArrangement = Arrangement.spacedBy(6.dp)
                ) {
                    analysis.entities.forEach { (key, value) ->
                        Box(
                            modifier = Modifier
                                .clip(RoundedCornerShape(6.dp))
                                .background(BgCard)
                                .border(1.dp, BorderSubtle, RoundedCornerShape(6.dp))
                                .padding(horizontal = 8.dp, vertical = 4.dp)
                        ) {
                            Text(
                                text = "$key: $value",
                                color = TextPrimary,
                                fontSize = 11.sp
                            )
                        }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // Confidence Metrics Bars
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(10.dp))
                .background(BgCardElevated)
                .border(1.dp, BorderSubtle, RoundedCornerShape(10.dp))
                .padding(12.dp)
        ) {
            Text(
                text = "STAGE CONFIDENCE BREAKDOWN",
                color = TextMuted,
                fontSize = 10.sp,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(8.dp))
            ConfidenceBarRow("Language", analysis.confidence.language, CyanAccent)
            Spacer(modifier = Modifier.height(6.dp))
            ConfidenceBarRow("Dialect", analysis.confidence.dialect, SkyBlue)
            Spacer(modifier = Modifier.height(6.dp))
            ConfidenceBarRow("Intent", analysis.confidence.intent, PurpleBadge)
        }
    }
}

@Composable
private fun LdmBadge(
    label: String,
    value: String,
    color: Color
) {
    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(8.dp))
            .background(color.copy(alpha = 0.12f))
            .border(1.dp, color.copy(alpha = 0.35f), RoundedCornerShape(8.dp))
            .padding(horizontal = 10.dp, vertical = 6.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                text = "$label: ",
                color = TextMuted,
                fontSize = 11.sp
            )
            Text(
                text = value,
                color = color,
                fontSize = 11.sp,
                fontWeight = FontWeight.Bold
            )
        }
    }
}

@Composable
private fun ConfidenceBarRow(
    title: String,
    score: Float,
    barColor: Color
) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = title,
            color = TextSecondary,
            fontSize = 11.sp,
            modifier = Modifier.width(68.dp)
        )
        LinearProgressIndicator(
            progress = { score },
            modifier = Modifier
                .weight(1f)
                .height(6.dp)
                .clip(RoundedCornerShape(3.dp)),
            color = barColor,
            trackColor = BgCard
        )
        Spacer(modifier = Modifier.width(8.dp))
        Text(
            text = "${(score * 100).toInt()}%",
            color = TextPrimary,
            fontSize = 11.sp,
            fontWeight = FontWeight.Medium,
            modifier = Modifier.width(36.dp)
        )
    }
}

