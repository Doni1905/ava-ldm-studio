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
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Check
import androidx.compose.material.icons.filled.ContentCopy
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.ava.ldm.model.LdmAnalysis
import ai.ava.ldm.ui.theme.BgCard
import ai.ava.ldm.ui.theme.BorderSubtle
import ai.ava.ldm.ui.theme.CodeBackground
import ai.ava.ldm.ui.theme.CyanAccent
import ai.ava.ldm.ui.theme.MintSuccess
import ai.ava.ldm.ui.theme.MonospaceCodeStyle
import ai.ava.ldm.ui.theme.TextMuted
import ai.ava.ldm.ui.theme.TextPrimary
import ai.ava.ldm.ui.theme.TextSecondary
import kotlinx.coroutines.delay

@Composable
fun HandoffSection(
    analysis: LdmAnalysis,
    modifier: Modifier = Modifier
) {
    val clipboardManager = LocalClipboardManager.current
    var copied by remember { mutableStateOf(false) }
    val handoffJson = remember(analysis) { analysis.toHandoffJson() }

    LaunchedEffect(copied) {
        if (copied) {
            delay(2000)
            copied = false
        }
    }

    Column(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(BgCard)
            .border(1.dp, BorderSubtle, RoundedCornerShape(16.dp))
            .padding(16.dp)
    ) {
        // Section Header with Copy Button
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "Local LLM Handoff Payload",
                    color = TextPrimary,
                    fontSize = 17.sp,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "Normalized linguistic understanding ready for Local LLM",
                    color = TextSecondary,
                    fontSize = 12.sp
                )
            }

            Button(
                onClick = {
                    clipboardManager.setText(AnnotatedString(handoffJson))
                    copied = true
                },
                shape = RoundedCornerShape(8.dp),
                colors = ButtonDefaults.buttonColors(
                    containerColor = if (copied) MintSuccess else CyanAccent.copy(alpha = 0.2f),
                    contentColor = if (copied) BgCard else CyanAccent
                ),
                contentPadding = androidx.compose.foundation.layout.PaddingValues(
                    horizontal = 12.dp,
                    vertical = 6.dp
                ),
                modifier = Modifier.height(34.dp)
            ) {
                Icon(
                    imageVector = if (copied) Icons.Default.Check else Icons.Default.ContentCopy,
                    contentDescription = null,
                    modifier = Modifier.height(14.dp).width(14.dp)
                )
                Spacer(modifier = Modifier.width(6.dp))
                Text(
                    text = if (copied) "Copied" else "Copy JSON",
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // Architectural Boundary Notice
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(8.dp))
                .background(CodeBackground)
                .border(1.dp, BorderSubtle, RoundedCornerShape(8.dp))
                .padding(10.dp)
        ) {
            Row(verticalAlignment = Alignment.Top) {
                Icon(
                    imageVector = Icons.Default.Info,
                    contentDescription = null,
                    tint = CyanAccent,
                    modifier = Modifier
                        .padding(top = 2.dp)
                        .height(14.dp)
                        .width(14.dp)
                )
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    text = "Architectural boundary: The LDM only normalizes dialectal Tamil/Tanglish and extracts intent. The LDM does NOT execute Android actions or generate conversational responses.",
                    color = TextMuted,
                    fontSize = 11.sp,
                    lineHeight = 15.sp
                )
            }
        }

        Spacer(modifier = Modifier.height(12.dp))

        // JSON Code Container
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(10.dp))
                .background(CodeBackground)
                .border(1.dp, BorderSubtle, RoundedCornerShape(10.dp))
                .padding(12.dp)
        ) {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .horizontalScroll(rememberScrollState())
            ) {
                Text(
                    text = handoffJson,
                    style = MonospaceCodeStyle
                )
            }
        }
    }
}

