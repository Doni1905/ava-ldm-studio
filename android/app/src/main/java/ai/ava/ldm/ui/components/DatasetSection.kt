package ai.ava.ldm.ui.components

import ai.ava.ldm.data.SyntheticDataset
import ai.ava.ldm.model.DatasetItem
import ai.ava.ldm.ui.theme.BgCard
import ai.ava.ldm.ui.theme.BgCardElevated
import ai.ava.ldm.ui.theme.BorderSubtle
import ai.ava.ldm.ui.theme.CyanAccent
import ai.ava.ldm.ui.theme.PurpleBadge
import ai.ava.ldm.ui.theme.TextMuted
import ai.ava.ldm.ui.theme.TextPrimary
import ai.ava.ldm.ui.theme.TextSecondary
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp

/** Browse the 30-item synthetic dataset; tapping an item runs it through the LDM. */
@Composable
fun DatasetSection(
    onSelectUtterance: (String) -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(BgCard)
            .border(1.dp, BorderSubtle, RoundedCornerShape(16.dp))
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        Text("Synthetic dataset", color = TextPrimary, fontSize = 18.sp, fontWeight = FontWeight.SemiBold)
        Text(
            "${SyntheticDataset.ITEMS.size} hand-written utterances (Tamil, Tanglish, English). " +
                "Tap one to analyse it in the Playground.",
            color = TextSecondary, fontSize = 13.sp
        )
        Spacer(Modifier.height(4.dp))
        SyntheticDataset.ITEMS.forEach { item -> DatasetRow(item, onSelectUtterance) }
    }
}

@Composable
private fun DatasetRow(item: DatasetItem, onSelect: (String) -> Unit) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(BgCardElevated)
            .clickable { onSelect(item.input) }
            .padding(12.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp)
    ) {
        Text(item.input, color = TextPrimary, fontSize = 15.sp, fontWeight = FontWeight.Medium)
        Text(item.normalized, color = TextSecondary, fontSize = 13.sp)
        Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
            Text(item.intent, color = CyanAccent, fontSize = 11.sp)
            Text(item.language, color = PurpleBadge, fontSize = 11.sp)
            Text(item.dialect, color = TextMuted, fontSize = 11.sp)
        }
    }
}
