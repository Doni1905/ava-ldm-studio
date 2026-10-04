package ai.ava.ldm.data

import ai.ava.ldm.model.DatasetItem

/**
 * Synthetic prototype dataset containing ~30 local utterances covering
 * Tamil, Tanglish, English, regional dialects, and slang across 10+ intents.
 */
object SyntheticDataset {

    val SAMPLE_UTTERANCES = listOf(
        "Dei nalaiku assignment submit panna remind pannu",
        "Machi inniku evening gym poga remind pannu",
        "Naalaikku kaalaila 6 maniku alarm vai",
        "Thambi ku oru message anuppu naan late ah varen nu",
        "Ayya nalaiku medicine saapida remind pannunga",
        "Friend ku call pottu kudu pa"
    )

    val ITEMS: List<DatasetItem> = listOf(
        DatasetItem(1, "Dei nalaiku assignment submit panna remind pannu", "Tanglish", "Chennai", true, "Remind me to submit my assignment tomorrow.", "CREATE_REMINDER"),
        DatasetItem(2, "Machi inniku evening gym poga remind pannu", "Tanglish", "Chennai", true, "Remind me to go to the gym this evening.", "CREATE_REMINDER"),
        DatasetItem(3, "Naalaikku kaalaila 6 maniku alarm vai", "Tamil (romanised)", "Standard", true, "Set an alarm for 6 in the morning tomorrow.", "SET_ALARM"),
        DatasetItem(4, "Amma ku call pannu", "Tanglish", "Standard", true, "Call mother.", "MAKE_CALL"),
        DatasetItem(5, "Thambi ku oru message anuppu naan late ah varen nu", "Tanglish", "Madurai", true, "Send a message to my brother that I will come late.", "SEND_MESSAGE"),
        DatasetItem(6, "Semma song ondru podu", "Tamil (romanised)", "Chennai", false, "Play a good song.", "PLAY_MUSIC"),
        DatasetItem(7, "Inniku weather eppadi iruku", "Tanglish", "Standard", true, "How is the weather today?", "CHECK_WEATHER"),
        DatasetItem(8, "WhatsApp open pannu da", "Tanglish", "Chennai", true, "Open WhatsApp.", "OPEN_APP"),
        DatasetItem(9, "Coimbatore ku vazhi kaatu", "Tamil (romanised)", "Kongu", false, "Show me directions to Coimbatore.", "NAVIGATE"),
        DatasetItem(10, "Volume konjam kammi pannu", "Tanglish", "Standard", true, "Reduce the volume a little.", "DEVICE_SETTING"),
        DatasetItem(11, "Anna nalaiku meeting iruku remind pannunga", "Tanglish", "Standard", true, "Remind me that there is a meeting tomorrow.", "CREATE_REMINDER"),
        DatasetItem(12, "Remind me to pay the electricity bill tomorrow", "English", "Standard", false, "Remind me to pay the electricity bill tomorrow.", "CREATE_REMINDER"),
        DatasetItem(13, "Set an alarm for 5:30 am", "English", "Standard", false, "Set an alarm for 5:30 am.", "SET_ALARM"),
        DatasetItem(14, "Appa ku phone pottu kudu", "Tamil (romanised)", "Nellai", false, "Call father.", "MAKE_CALL"),
        DatasetItem(15, "Ilayaraja paatu podu la", "Tamil (romanised)", "Kongu", false, "Play Ilayaraja song.", "PLAY_MUSIC"),
        DatasetItem(16, "Naalaikku mazhai varuma", "Tamil (romanised)", "Standard", false, "Will it rain tomorrow?", "CHECK_WEATHER"),
        DatasetItem(17, "Camera open pannu sikkiram", "Tanglish", "Standard", true, "Open the camera quickly.", "OPEN_APP"),
        DatasetItem(18, "Office ku route sollu bruh", "Tanglish", "Chennai", true, "Show me the route to the office.", "NAVIGATE"),
        DatasetItem(19, "Brightness konjam koothu", "Tanglish", "Standard", true, "Increase the brightness a little.", "DEVICE_SETTING"),
        DatasetItem(20, "Ava enna panra ipo", "Tamil (romanised)", "Chennai", false, "What are you doing now?", "SMALL_TALK"),
        DatasetItem(21, "Vanakkam AVA eppadi iruka", "Tamil (romanised)", "Standard", false, "Hello AVA, how are you?", "SMALL_TALK"),
        DatasetItem(22, "Bangalore population evlo nu search pannu", "Tanglish", "Standard", true, "Search how much the population of Bangalore is.", "SEARCH_INFO"),
        DatasetItem(23, "Tomorrow morning ku 7 maniku alarm vai da", "Tanglish", "Chennai", true, "Set an alarm for 7 tomorrow morning.", "SET_ALARM"),
        DatasetItem(24, "Sister ku message anuppu", "Tanglish", "Standard", true, "Send a message to my sister.", "SEND_MESSAGE"),
        DatasetItem(25, "Konjam music podu please", "Tanglish", "Standard", true, "Play some music.", "PLAY_MUSIC"),
        DatasetItem(26, "Ayya nalaiku medicine saapida remind pannunga", "Tanglish", "Kongu", true, "Remind me to take medicine tomorrow.", "CREATE_REMINDER"),
        DatasetItem(27, "Chennai la traffic eppadi iruku nu sollu", "Tanglish", "Chennai", true, "Tell me how the traffic in Chennai is.", "SEARCH_INFO"),
        DatasetItem(28, "Wifi off pannu", "Tanglish", "Standard", true, "Turn off the wifi.", "DEVICE_SETTING"),
        DatasetItem(29, "Friend ku call pottu kudu pa", "Tanglish", "Nellai", true, "Call my friend.", "MAKE_CALL"),
        DatasetItem(30, "Play some Tamil songs please", "English", "Standard", false, "Play some Tamil songs.", "PLAY_MUSIC")
    )
}

