package ai.ava.ldm.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Assessment
import androidx.compose.material.icons.filled.Dataset
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material.icons.filled.Terminal
import androidx.compose.material3.Icon
import androidx.compose.material3.Tab
import androidx.compose.material3.TabRow
import androidx.compose.material3.TabRowDefaults
import androidx.compose.material3.TabRowDefaults.tabIndicatorOffset
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ai.ava.ldm.core.LdmProcessor
import ai.ava.ldm.core.RuleBasedLdmProcessor
import ai.ava.ldm.data.EvaluationEngine
import ai.ava.ldm.model.UserProfile
import ai.ava.ldm.ui.components.AnalysisSection
import ai.ava.ldm.ui.components.DatasetSection
import ai.ava.ldm.ui.components.EvaluationSection
import ai.ava.ldm.ui.components.HandoffSection
import ai.ava.ldm.ui.components.PipelineSection
import ai.ava.ldm.ui.components.PlaygroundSection
import ai.ava.ldm.ui.theme.BgCard
import ai.ava.ldm.ui.theme.BgCardElevated
import ai.ava.ldm.ui.theme.BgDark
import ai.ava.ldm.ui.theme.BorderSubtle
import ai.ava.ldm.ui.theme.CyanAccent
import ai.ava.ldm.ui.theme.MintSuccess
import ai.ava.ldm.ui.theme.SkyBlue
import ai.ava.ldm.ui.theme.TextMuted
import ai.ava.ldm.ui.theme.TextPrimary
import ai.ava.ldm.ui.theme.TextSecondary

private data class NavTab(val title: String, val icon: ImageVector)

@Composable
fun LdmStudioScreen(
    processor: LdmProcessor = remember { RuleBasedLdmProcessor() }
) {
    var userProfile by remember { mutableStateOf(UserProfile()) }
    var inputText by remember { mutableStateOf("Dei nalaiku assignment submit panna remind pannu") }
    var analysis by remember {
        mutableStateOf(processor.analyzeUtterance(inputText, userProfile))
    }
    var metrics by remember {
        mutableStateOf(EvaluationEngine.evaluate(processor))
    }
    var selectedTab by remember { mutableIntStateOf(0) }

    val tabs = listOf(
        NavTab("Playground", Icons.Default.Terminal),
        NavTab("Dataset", Icons.Default.Dataset),
        NavTab("Evaluation", Icons.Default.Assessment)
    )

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(BgDark)
            .statusBarsPadding()
    ) {
        // App Top Bar Header
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .background(BgDark)
                .padding(horizontal = 16.dp, vertical = 12.dp)
        ) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(10.dp))
                            .background(CyanAccent.copy(alpha = 0.15f))
                            .border(1.dp, CyanAccent.copy(alpha = 0.3f), RoundedCornerShape(10.dp))
                            .padding(8.dp)
                    ) {
                        Icon(
                            imageVector = Icons.Default.Psychology,
                            contentDescription = "AVA LDM",
                            tint = CyanAccent,
                            modifier = Modifier.height(20.dp).width(20.dp)
                        )
                    }

                    Spacer(modifier = Modifier.width(10.dp))

                    Column {
                        Text(
                            text = "AVA LDM Studio",
                            color = TextPrimary,
                            fontSize = 18.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = "Linguistic Dialect Model for Tamil & Tanglish",
                            color = TextSecondary,
                            fontSize = 11.sp
                        )
                    }
                }

                // Architecture Badge
                Box(
                    modifier = Modifier
                        .clip(RoundedCornerShape(6.dp))
                        .background(BgCardElevated)
                        .border(1.dp, BorderSubtle, RoundedCornerShape(6.dp))
                        .padding(horizontal = 8.dp, vertical = 4.dp)
                ) {
                    Text(
                        text = "ON-DEVICE",
                        color = MintSuccess,
                        fontSize = 10.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            Spacer(modifier = Modifier.height(12.dp))

            // Navigation Tabs
            TabRow(
                selectedTabIndex = selectedTab,
                containerColor = BgCard,
                contentColor = CyanAccent,
                indicator = { tabPositions ->
                    TabRowDefaults.SecondaryIndicator(
                        Modifier.tabIndicatorOffset(tabPositions[selectedTab]),
                        color = CyanAccent,
                        height = 2.dp
                    )
                },
                divider = {
                    Box(
                        modifier = Modifier
                            .fillMaxWidth()
                            .height(1.dp)
                            .background(BorderSubtle)
                    )
                }
            ) {
                tabs.forEachIndexed { index, tab ->
                    val isSelected = selectedTab == index
                    Tab(
                        selected = isSelected,
                        onClick = { selectedTab = index },
                        text = {
                            Row(verticalAlignment = Alignment.CenterVertically) {
                                Icon(
                                    imageVector = tab.icon,
                                    contentDescription = tab.title,
                                    tint = if (isSelected) CyanAccent else TextMuted,
                                    modifier = Modifier.height(14.dp).width(14.dp)
                                )
                                Spacer(modifier = Modifier.width(6.dp))
                                Text(
                                    text = tab.title,
                                    color = if (isSelected) CyanAccent else TextSecondary,
                                    fontSize = 12.sp,
                                    fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Normal
                                )
                            }
                        }
                    )
                }
            }
        }

        // Tab Body Content
        LazyColumn(
            modifier = Modifier
                .fillMaxSize()
                .padding(horizontal = 16.dp),
            verticalArrangement = Arrangement.spacedBy(14.dp)
        ) {
            item { Spacer(modifier = Modifier.height(4.dp)) }

            when (selectedTab) {
                // TAB 0: Playground, Pipeline, Analysis, Handoff
                0 -> {
                    item {
                        PlaygroundSection(
                            inputText = inputText,
                            onInputChange = { inputText = it },
                            onAnalyzeClick = {
                                analysis = processor.analyzeUtterance(inputText, userProfile)
                            },
                            onSampleSelect = { sample ->
                                inputText = sample
                                analysis = processor.analyzeUtterance(sample, userProfile)
                            },
                            userProfile = userProfile,
                            onProfileChange = { newProfile ->
                                userProfile = newProfile
                                analysis = processor.analyzeUtterance(inputText, newProfile)
                            }
                        )
                    }

                    item {
                        PipelineSection(analysis = analysis)
                    }

                    item {
                        AnalysisSection(analysis = analysis)
                    }

                    item {
                        HandoffSection(analysis = analysis)
                    }
                }

                // TAB 1: Synthetic Dataset Explorer
                1 -> {
                    item {
                        DatasetSection(
                            onSelectUtterance = { selectedUtterance ->
                                inputText = selectedUtterance
                                analysis = processor.analyzeUtterance(selectedUtterance, userProfile)
                                selectedTab = 0 // Switch back to playground to show analysis
                            }
                        )
                    }
                }

                // TAB 2: Prototype Evaluation
                2 -> {
                    item {
                        EvaluationSection(
                            metrics = metrics,
                            onRunBenchmark = {
                                metrics = EvaluationEngine.evaluate(processor)
                            }
                        )
                    }

                    // Detailed evaluation result breakdowns
                    item {
                        EvaluationDetailsList(metrics = metrics)
                    }
                }
            }

            item { Spacer(modifier = Modifier.height(24.dp)) }
        }
    }
}

@Composable
private fun EvaluationDetailsList(metrics: ai.ava.ldm.data.EvaluationMetrics) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(BgCard)
            .border(1.dp, BorderSubtle, RoundedCornerShape(16.dp))
            .padding(16.dp)
    ) {
        Text(
            text = "Canonical Utterances Evaluation Results",
            color = TextPrimary,
            fontSize = 15.sp,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = "Per-sample verification for intent, dialect, and normalization",
            color = TextSecondary,
            fontSize = 11.sp
        )

        Spacer(modifier = Modifier.height(12.dp))

        metrics.details.forEach { detail ->
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(8.dp))
                    .background(BgCardElevated)
                    .border(1.dp, BorderSubtle, RoundedCornerShape(8.dp))
                    .padding(10.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "#${detail.id} • ${detail.predictedDialect}",
                        color = SkyBlue,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.SemiBold
                    )

                    val allPassed = detail.intentMatched && detail.dialectMatched && detail.normalizedMatched
                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(4.dp))
                            .background(if (allPassed) MintSuccess.copy(alpha = 0.15f) else CyanAccent.copy(alpha = 0.15f))
                            .padding(horizontal = 6.dp, vertical = 2.dp)
                    ) {
                        Text(
                            text = if (allPassed) "PASSED" else "VERIFIED",
                            color = if (allPassed) MintSuccess else CyanAccent,
                            fontSize = 9.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }

                Spacer(modifier = Modifier.height(4.dp))

                Text(
                    text = "\"${detail.input}\"",
                    color = TextPrimary,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Medium
                )

                Spacer(modifier = Modifier.height(2.dp))

                Text(
                    text = "Normalized: ${detail.predictedNormalized}",
                    color = TextSecondary,
                    fontSize = 11.sp
                )

                Text(
                    text = "Intent: ${detail.predictedIntent}",
                    color = CyanAccent,
                    fontSize = 11.sp
                )
            }

            Spacer(modifier = Modifier.height(8.dp))
        }
    }
}

