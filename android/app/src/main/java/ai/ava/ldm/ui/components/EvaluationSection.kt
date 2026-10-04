package ai.ava.ldm.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Assessment
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
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
import ai.ava.ldm.data.EvaluationMetrics
import ai.ava.ldm.ui.theme.AmberWarning
import ai.ava.ldm.ui.theme.BgCard
import ai.ava.ldm.ui.theme.BgCardElevated
import ai.ava.ldm.ui.theme.BorderSubtle
import ai.ava.ldm.ui.theme.CyanAccent
import ai.ava.ldm.ui.theme.MintSuccess
import ai.ava.ldm.ui.theme.SkyBlue
import ai.ava.ldm.ui.theme.TextMuted
import ai.ava.ldm.ui.theme.TextPrimary
import ai.ava.ldm.ui.theme.TextSecondary
import java.util.Locale

@Composable
fun EvaluationSection(
    metrics: EvaluationMetrics,
    onRunBenchmark: () -> Unit,
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
        // Header
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Icon(
                        imageVector = Icons.Default.Assessment,
                        contentDescription = null,
                        tint = MintSuccess,
                        modifier = Modifier.height(18.dp).width(18.dp)
                    )
                    Spacer(modifier = Modifier.width(6.dp))
                    Text(
                        text = "Synthetic Prototype Evaluation",
                        color = TextPrimary,
                        fontSize = 17.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
                Text(
                    text = "Benchmark metrics across 30 test utterances",
                    color = TextSecondary,
                    fontSize = 12.sp
                )
            }

            Button(
                onClick = onRunBenchmark,
                shape = RoundedCornerShape(8.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = MintSuccess.copy(alpha = 0.2f),
                    contentColor = MintSuccess
                ),
                contentPadding = androidx.compose.foundation.layout.PaddingValues(
                    horizontal = 10.dp,
                    vertical = 6.dp
                ),
                modifier = Modifier.height(34.dp)
            ) {
                Icon(
                    imageVector = Icons.Default.Refresh,
                    contentDescription = null,
                    modifier = Modifier.height(14.dp).width(14.dp)
                )
                Spacer(modifier = Modifier.width(4.dp))
                Text(
                    text = "Re-run",
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }

        Spacer(modifier = Modifier.height(14.dp))

        // Metrics Grid (2x2)
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            MetricCard(
                modifier = Modifier.weight(1f),
                title = "NORMALIZATION",
                value = String.format(Locale.US, "%.1f%%", metrics.normalizationAccuracy),
                subtitle = "Semantic preservation",
                color = MintSuccess
            )
            MetricCard(
                modifier = Modifier.weight(1f),
                title = "INTENT ACCURACY",
                value = String.format(Locale.US, "%.1f%%", metrics.intentAccuracy),
                subtitle = "Classification match",
                color = CyanAccent
            )
        }

        Spacer(modifier = Modifier.height(10.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            MetricCard(
                modifier = Modifier.weight(1f),
                title = "DIALECT ACCURACY",
                value = String.format(Locale.US, "%.1f%%", metrics.dialectAccuracy),
                subtitle = "Regional Tamil match",
                color = SkyBlue
            )
            MetricCard(
                modifier = Modifier.weight(1f),
                title = "CODE-MIX ACCURACY",
                value = String.format(Locale.US, "%.1f%%", metrics.codeMixAccuracy),
                subtitle = "Mixed detection",
                color = AmberWarning
            )
        }

        Spacer(modifier = Modifier.height(12.dp))

        // Latency Summary Banner
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(10.dp))
                .background(BgCardElevated)
                .border(1.dp, BorderSubtle, RoundedCornerShape(10.dp))
                .padding(12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Column {
                    Text(
                        text = "MEAN ON-DEVICE LATENCY",
                        color = TextMuted,
                        fontSize = 10.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = String.format(Locale.US, "%.3f ms per utterance", metrics.avgLatencyMs),
                        color = MintSuccess,
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold
                    )
                }

                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(6.dp))
                        .background(MintSuccess.copy(alpha = 0.15f))
                        .padding(horizontal = 8.dp, vertical = 4.dp)
                ) {
                    Text(
                        text = "Sub-millisecond",
                        color = MintSuccess,
                        fontSize = 10.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(10.dp))

        Text(
            text = "Note: Evaluated on synthetic prototype benchmark test suite. Real speech Whisper ASR + Wav2Vec2 evaluations are documented in project results/evaluation_report.md.",
            color = TextMuted,
            fontSize = 11.sp,
            lineHeight = 15.sp
        )
    }
}

@Composable
private fun MetricCard(
    modifier: Modifier = Modifier,
    title: String,
    value: String,
    subtitle: String,
    color: Color
) {
    Column(
        modifier = modifier
            .clip(RoundedCornerShape(10.dp))
            .background(BgCardElevated)
            .border(1.dp, BorderSubtle, RoundedCornerShape(10.dp))
            .padding(12.dp)
    ) {
        Text(
            text = title,
            color = TextMuted,
            fontSize = 10.sp,
            fontWeight = FontWeight.Bold
        )
        Spacer(modifier = Modifier.height(4.dp))
        Text(
            text = value,
            color = color,
            fontSize = 20.sp,
            fontWeight = FontWeight.Bold
        )
        Spacer(modifier = Modifier.height(2.dp))
        Text(
            text = subtitle,
            color = TextSecondary,
            fontSize = 10.sp,
            maxLines = 1
        )
    }
}

