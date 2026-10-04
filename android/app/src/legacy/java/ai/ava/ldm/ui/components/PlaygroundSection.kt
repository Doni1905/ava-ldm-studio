package ai.ava.ldm.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
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
import androidx.compose.material.icons.filled.Clear
import androidx.compose.material.icons.filled.PlayArrow
import androidx.compose.material.icons.filled.Tune
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.ava.ldm.data.SyntheticDataset
import ai.ava.ldm.model.UserProfile
import ai.ava.ldm.ui.theme.BgCard
import ai.ava.ldm.ui.theme.BgCardElevated
import ai.ava.ldm.ui.theme.BorderHighlight
import ai.ava.ldm.ui.theme.BorderSubtle
import ai.ava.ldm.ui.theme.CyanAccent
import ai.ava.ldm.ui.theme.TextMuted
import ai.ava.ldm.ui.theme.TextPrimary
import ai.ava.ldm.ui.theme.TextSecondary

@Composable
fun PlaygroundSection(
    inputText: String,
    onInputChange: (String) -> Unit,
    onAnalyzeClick: () -> Unit,
    onSampleSelect: (String) -> Unit,
    userProfile: UserProfile,
    onProfileChange: (UserProfile) -> Unit,
    modifier: Modifier = Modifier
) {
    var showProfileSettings by remember { mutableStateOf(false) }

    Column(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(BgCard)
            .border(1.dp, BorderSubtle, RoundedCornerShape(16.dp))
            .padding(16.dp)
    ) {
        // Section Header
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1f)) {
                Text(
                    text = "LDM Playground",
                    color = TextPrimary,
                    fontSize = 17.sp,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = "Type Tamil/Tanglish text or choose a sample",
                    color = TextSecondary,
                    fontSize = 12.sp
                )
            }

            IconButton(onClick = { showProfileSettings = !showProfileSettings }) {
                Icon(
                    imageVector = Icons.Default.Tune,
                    contentDescription = "Personalization Profile",
                    tint = if (showProfileSettings) CyanAccent else TextMuted
                )
            }
        }

        // Profile Quick Setting Bar (expandable)
        if (showProfileSettings) {
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
                    text = "Personalization Context (Dialect Preference)",
                    color = CyanAccent,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.SemiBold
                )
                Spacer(modifier = Modifier.height(8.dp))
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .horizontalScroll(rememberScrollState()),
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    val dialects = listOf("Standard", "Chennai", "Kongu", "Madurai", "Nellai")
                    for (dialect in dialects) {
                        val isSelected = userProfile.preferredDialect == dialect
                        Box(
                            modifier = Modifier
                                .clip(RoundedCornerShape(8.dp))
                                .background(if (isSelected) CyanAccent.copy(alpha = 0.2f) else BgCard)
                                .border(
                                    1.dp,
                                    if (isSelected) CyanAccent else BorderSubtle,
                                    RoundedCornerShape(8.dp)
                                )
                                .clickable {
                                    onProfileChange(userProfile.copy(preferredDialect = dialect))
                                }
                                .padding(horizontal = 10.dp, vertical = 6.dp)
                        ) {
                            Text(
                                text = dialect,
                                color = if (isSelected) CyanAccent else TextSecondary,
                                fontSize = 11.sp,
                                fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal
                            )
                        }
                    }
                }
            }
        }

        Spacer(modifier = Modifier.height(14.dp))

        // Sample Utterances Row
        Text(
            text = "SAMPLE UTTERANCES",
            color = TextMuted,
            fontSize = 10.sp,
            fontWeight = FontWeight.Bold,
            letterSpacing = 1.sp
        )
        Spacer(modifier = Modifier.height(6.dp))

        Row(
            modifier = Modifier
                .fillMaxWidth()
                .horizontalScroll(rememberScrollState()),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            SyntheticDataset.SAMPLE_UTTERANCES.forEach { sample ->
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(20.dp))
                        .background(BgCardElevated)
                        .border(1.dp, BorderSubtle, RoundedCornerShape(20.dp))
                        .clickable { onSampleSelect(sample) }
                        .padding(horizontal = 12.dp, vertical = 7.dp)
                ) {
                    Text(
                        text = sample,
                        color = TextPrimary,
                        fontSize = 12.sp,
                        maxLines = 1
                    )
                }
            }
        }

        Spacer(modifier = Modifier.height(14.dp))

        // Input Field
        OutlinedTextField(
            value = inputText,
            onValueChange = onInputChange,
            modifier = Modifier.fillMaxWidth(),
            placeholder = {
                Text(
                    text = "e.g. Dei nalaiku assignment submit panna remind pannu",
                    color = TextMuted,
                    fontSize = 13.sp
                )
            },
            trailingIcon = {
                if (inputText.isNotEmpty()) {
                    IconButton(onClick = { onInputChange("") }) {
                        Icon(
                            imageVector = Icons.Default.Clear,
                            contentDescription = "Clear",
                            tint = TextMuted
                        )
                    }
                }
            },
            colors = OutlinedTextFieldDefaults.colors(
                focusedContainerColor = BgCardElevated,
                unfocusedContainerColor = BgCardElevated,
                focusedBorderColor = BorderHighlight,
                unfocusedBorderColor = BorderSubtle,
                focusedTextColor = TextPrimary,
                unfocusedTextColor = TextPrimary,
                cursorColor = CyanAccent
            ),
            shape = RoundedCornerShape(12.dp),
            minLines = 2,
            maxLines = 4
        )

        Spacer(modifier = Modifier.height(12.dp))

        // Analyze Action Button
        Button(
            onClick = onAnalyzeClick,
            modifier = Modifier
                .fillMaxWidth()
                .height(46.dp),
            shape = RoundedCornerShape(12.dp),
            colors = ButtonDefaults.buttonColors(
                containerColor = CyanAccent,
                contentColor = BgCard
            )
        ) {
            Icon(
                imageVector = Icons.Default.PlayArrow,
                contentDescription = null
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = "Run LDM Understanding Pipeline",
                fontWeight = FontWeight.Bold,
                fontSize = 14.sp
            )
        }
    }
}

