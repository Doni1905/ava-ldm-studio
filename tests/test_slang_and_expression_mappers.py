"""
tests/test_slang_and_expression_mappers.py
===========================================
Tests for:
  - src/linguistic/slang_mapper.py      (SlangMapper)
  - src/linguistic/expression_mapper.py (ExpressionMapper)

Test strategy
-------------
1. **Per-mapping tests** — every row in every CSV is tested to confirm:
   a. The informal form is in the vocabulary (can be looked up)
   b. lookup() returns matched=True
   c. The returned normalized value matches the CSV
   d. The category matches the CSV

2. **Filler tests** — all entries with normalized="<FILLER>" are tested
   to confirm is_filler=True and apply() drops them.

3. **Word-boundary tests** — confirm substring NON-matches (critical
   contract: "pannu" in "appannu" must NOT match).

4. **Apply tests** — confirm apply() produces correct normalized output
   and drops fillers.

5. **Expression tests** — every row in personal_expressions.csv:
   a. Is in the vocabulary
   b. lookup() returns the correct normalized form
   c. map() on the bare expression returns the normalized form

6. **Longest-match tests** — confirm longer expressions take priority
   over shorter ones when they share a prefix.

7. **No-fabrication tests** — confirm unknown tokens pass through unchanged.

8. **Semantic isolation tests** — confirm unrelated tokens alongside a
   matched token are not altered.

Run:
    python -m pytest tests/test_slang_and_expression_mappers.py -v
    python tests/test_slang_and_expression_mappers.py
"""

from __future__ import annotations

import csv
import io
import sys
import unittest
from pathlib import Path
from typing import List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.linguistic.slang_mapper import (
    SlangMapper, MappingResult, LexCategory, FILLER_SENTINEL
)
from src.linguistic.expression_mapper import (
    ExpressionMapper, ExpressionEntry, ExpressionMapResult
)

# ---------------------------------------------------------------------------
# CSV paths
# ---------------------------------------------------------------------------
_SLANG_CSV    = _ROOT / "data" / "linguistic" / "slang_dictionary.csv"
_INFORMAL_CSV = _ROOT / "data" / "linguistic" / "informal_dictionary.csv"
_EXPR_CSV     = _ROOT / "data" / "linguistic" / "personal_expressions.csv"


def _load_csv_rows(path: Path) -> List[dict]:
    """Load all non-empty, non-comment rows from a CSV."""
    rows = []
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            key = list(row.values())[0].strip()
            if key and not key.startswith("#"):
                rows.append(row)
    return rows


# ============================================================================
# Fixtures
# ============================================================================

def _make_mapper() -> SlangMapper:
    return SlangMapper.default(project_root=_ROOT)

def _make_expr_mapper() -> ExpressionMapper:
    return ExpressionMapper.default(project_root=_ROOT)


# ============================================================================
# 1. SlangMapper — CSV loading
# ============================================================================

class TestSlangMapperLoading(unittest.TestCase):

    def setUp(self):
        self.sm = _make_mapper()

    def test_slang_csv_exists(self):
        self.assertTrue(_SLANG_CSV.exists(), f"Missing: {_SLANG_CSV}")

    def test_informal_csv_exists(self):
        self.assertTrue(_INFORMAL_CSV.exists(), f"Missing: {_INFORMAL_CSV}")

    def test_entries_loaded(self):
        self.assertGreater(len(self.sm.all_entries()), 0)

    def test_vocabulary_non_empty(self):
        self.assertGreater(len(self.sm.vocabulary()), 0)

    def test_all_entries_have_normalized(self):
        for entry in self.sm.all_entries():
            self.assertIsInstance(entry.normalized, str)

    def test_all_entries_have_category(self):
        for entry in self.sm.all_entries():
            self.assertIsInstance(entry.category, str)
            self.assertGreater(len(entry.category), 0)


# ============================================================================
# 2. SlangMapper — Per-mapping coverage (parametric over CSV rows)
# ============================================================================

class TestSlangMapperAllEntries(unittest.TestCase):
    """Every row in slang_dictionary.csv and informal_dictionary.csv is tested."""

    @classmethod
    def setUpClass(cls):
        cls.sm = _make_mapper()
        cls.slang_rows    = _load_csv_rows(_SLANG_CSV)
        cls.informal_rows = _load_csv_rows(_INFORMAL_CSV)
        cls.all_rows      = cls.slang_rows + cls.informal_rows

    def test_every_slang_entry_in_vocabulary(self):
        missing = []
        for row in self.slang_rows:
            key = row["informal"].strip().lower()
            if key not in self.sm.vocabulary():
                missing.append(key)
        self.assertEqual(missing, [], f"Slang entries not in vocabulary: {missing}")

    def test_every_informal_entry_in_vocabulary(self):
        missing = []
        for row in self.informal_rows:
            key = row["informal"].strip().lower()
            if key not in self.sm.vocabulary():
                missing.append(key)
        self.assertEqual(missing, [], f"Informal entries not in vocabulary: {missing}")

    def test_every_entry_lookup_matched(self):
        failures = []
        for row in self.all_rows:
            key = row["informal"].strip().lower()
            result = self.sm.lookup(key)
            if not result.matched:
                failures.append(key)
        self.assertEqual(failures, [],
            f"lookup() returned matched=False for: {failures}")

    def test_every_entry_normalized_correct(self):
        """Every lookup returns the normalized value matching the CSV."""
        failures = []
        for row in self.all_rows:
            key        = row["informal"].strip().lower()
            expected   = row["normalized"].strip()
            result     = self.sm.lookup(key)
            # Filler entries: display_normalized is ""
            if expected == FILLER_SENTINEL:
                if result.normalized != "":
                    failures.append((key, "expected filler", result.normalized))
            else:
                if result.normalized != expected:
                    failures.append((key, expected, result.normalized))
        self.assertEqual(failures, [],
            f"Normalized value mismatches: {failures[:5]}")

    def test_every_entry_category_correct(self):
        failures = []
        for row in self.all_rows:
            key      = row["informal"].strip().lower()
            expected = row["category"].strip().upper()
            result   = self.sm.lookup(key)
            if result.category != expected:
                failures.append((key, expected, result.category))
        self.assertEqual(failures, [],
            f"Category mismatches: {failures[:5]}")

    def test_filler_entries_flagged(self):
        """All CSV rows with normalized='<FILLER>' must have is_filler=True."""
        failures = []
        for row in self.all_rows:
            if row["normalized"].strip() == FILLER_SENTINEL:
                key    = row["informal"].strip().lower()
                result = self.sm.lookup(key)
                if not result.is_filler:
                    failures.append(key)
        self.assertEqual(failures, [],
            f"Entries that should be fillers but aren't: {failures}")


# ============================================================================
# 3. SlangMapper — Specific known mappings
# ============================================================================

class TestSlangMapperKnownMappings(unittest.TestCase):

    def setUp(self):
        self.sm = _make_mapper()

    def _check(self, token: str, expected_norm: str, expected_cat: str):
        r = self.sm.lookup(token)
        self.assertTrue(r.matched, f"'{token}' not matched")
        self.assertEqual(r.normalized, expected_norm,
            f"'{token}': expected norm '{expected_norm}', got '{r.normalized}'")
        self.assertEqual(r.category, expected_cat,
            f"'{token}': expected cat '{expected_cat}', got '{r.category}'")

    # --- Verbs ---
    def test_pannu(self):     self._check("pannu",    "do",       "VERB")
    def test_pannunga(self):  self._check("pannunga", "please do","VERB")
    def test_sollu(self):     self._check("sollu",    "tell",     "VERB")
    def test_sollunga(self):  self._check("sollunga", "tell me",  "VERB")
    def test_paaru(self):     self._check("paaru",    "look / check", "VERB")
    def test_paarunga(self):  self._check("paarunga", "please check", "VERB")
    def test_vaa(self):       self._check("vaa",      "come",     "VERB")
    def test_poo(self):       self._check("poo",      "go",       "VERB")
    def test_kudu(self):      self._check("kudu",     "give",     "VERB")
    def test_podu(self):      self._check("podu",     "play / put","VERB")
    def test_saapidu(self):   self._check("saapidu",  "eat",      "VERB")

    # --- Fillers ---
    def test_dei_is_filler(self):
        r = self.sm.lookup("dei")
        self.assertTrue(r.is_filler)
        self.assertEqual(r.normalized, "")

    def test_da_is_filler(self):
        r = self.sm.lookup("da")
        self.assertTrue(r.is_filler)

    def test_machi_is_filler(self):
        r = self.sm.lookup("machi")
        self.assertTrue(r.is_filler)

    def test_la_is_filler(self):
        r = self.sm.lookup("la")
        self.assertTrue(r.is_filler)

    def test_pa_is_filler(self):
        r = self.sm.lookup("pa")
        self.assertTrue(r.is_filler)

    def test_nu_is_filler(self):
        r = self.sm.lookup("nu")
        self.assertTrue(r.is_filler)

    # --- Temporal ---
    def test_nalaiku(self):    self._check("nalaiku",   "tomorrow",        "TEMPORAL")
    def test_naalaikku(self):  self._check("naalaikku", "tomorrow",        "TEMPORAL")
    def test_inniku(self):     self._check("inniku",    "today",           "TEMPORAL")
    def test_ippo(self):       self._check("ippo",      "now",             "TEMPORAL")
    def test_ipo(self):        self._check("ipo",       "now",             "TEMPORAL")
    def test_raathiri(self):   self._check("raathiri",  "tonight",         "TEMPORAL")
    def test_kaalaila(self):   self._check("kaalaila",  "in the morning",  "TEMPORAL")

    # --- Intensifiers ---
    def test_romba(self):      self._check("romba",     "very",            "INTENSIFIER")
    def test_bayangara(self):  self._check("bayangara", "extremely",       "INTENSIFIER")
    def test_semma(self):      self._check("semma",     "great / very good", "INTENSIFIER")
    def test_nalla(self):      self._check("nalla",     "good",            "ADJECTIVE")
    def test_konjam(self):     self._check("konjam",    "a little / some", "QUANTIFIER")

    # --- Person / People ---
    def test_amma(self):  self._check("amma",  "mother",     "PRONOUN")
    def test_appa(self):  self._check("appa",  "father",     "PRONOUN")

    # --- Pronouns / Questions ---
    def test_naan(self):   self._check("naan",  "I",               "PRONOUN")
    def test_nee(self):    self._check("nee",   "you",             "PRONOUN")
    def test_enna(self):   self._check("enna",  "what",            "QUESTION")
    def test_evlo(self):   self._check("evlo",  "how much / how many", "QUESTION")
    def test_eppo(self):   self._check("eppo",  "when",            "QUESTION")
    def test_eppadi(self): self._check("eppadi","how",             "QUESTION")

    # --- Greetings ---
    def test_vanakkam(self): self._check("vanakkam", "hello", "GREETING")
    def test_bye(self):      self._check("bye",      "goodbye","GREETING")

    # --- Affirmative / Negation ---
    def test_seri(self):    self._check("seri",    "okay / alright",  "AFFIRMATIVE")
    def test_aama(self):    self._check("aama",    "yes",             "AFFIRMATIVE")
    def test_illa(self):    self._check("illa",    "no",              "NEGATION")
    def test_illai(self):   self._check("illai",   "no / not",        "NEGATION")
    def test_vendam(self):  self._check("vendam",  "don't want / don't", "NEGATION")

    # --- Contractions ---
    def test_mateen(self):      self._check("mateen",    "will not",           "CONTRACTION")
    def test_matten(self):      self._check("matten",    "will not",           "CONTRACTION")
    def test_theriyathu(self):  self._check("theriyathu","I don't know",       "CONTRACTION")
    def test_theriyum(self):    self._check("theriyum",  "I know",             "CONTRACTION")
    def test_puriyuthu(self):   self._check("puriyuthu", "I understand",       "CONTRACTION")
    def test_puriyala(self):    self._check("puriyala",  "I don't understand", "CONTRACTION")
    def test_mudiyathu(self):   self._check("mudiyathu", "cannot",             "CONTRACTION")

    # --- Informal verbs ---
    def test_varen(self):    self._check("varen",    "coming",  "INFORMAL_VERB")
    def test_pogiren(self):  self._check("pogiren",  "going",   "INFORMAL_VERB")
    def test_pokiren(self):  self._check("pokiren",  "going",   "INFORMAL_VERB")
    def test_vandhen(self):  self._check("vandhen",  "came",    "INFORMAL_VERB")
    def test_irukken(self):  self._check("irukken",  "I am (here)", "INFORMAL_VERB")


# ============================================================================
# 4. SlangMapper — Word-boundary contract (no substring matches)
# ============================================================================

class TestSlangMapperWordBoundary(unittest.TestCase):
    """
    The mapper must NEVER match tokens as substrings of other words.
    'pannu' in 'pannunga' must not match as 'pannu'.
    """

    def setUp(self):
        self.sm = _make_mapper()

    def test_pannu_not_in_pannunga(self):
        # 'pannunga' should be its own entry, not matched as 'pannu'
        r = self.sm.lookup("pannunga")
        self.assertEqual(r.normalized, "please do")

    def test_pannu_not_matching_inside_longer_word(self):
        # apply() on a text where 'pannu' is part of a longer word
        # The token "pannuvaraa" is not in dict, so it passes through
        out = self.sm.apply("pannuvaraa enna")
        self.assertIn("pannuvaraa", out)  # passed through
        self.assertNotIn("do varaa", out)  # NOT split-matched

    def test_la_not_in_nalaiku(self):
        # 'la' is a filler, but 'nalaiku' contains 'la' as substring
        r = self.sm.lookup("nalaiku")
        self.assertEqual(r.normalized, "tomorrow")  # correct entry, not filler

    def test_da_not_in_vandha(self):
        # 'da' filler should not match inside 'vandha'
        r = self.sm.lookup("vandha")
        self.assertTrue(r.matched)  # 'vandha' has its own entry
        self.assertFalse(r.is_filler)

    def test_unknown_token_passthrough(self):
        r = self.sm.lookup("xyzfakeword123")
        self.assertFalse(r.matched)
        self.assertEqual(r.normalized, "xyzfakeword123")

    def test_partial_word_not_matched(self):
        r = self.sm.lookup("solluv")  # not in dictionary
        self.assertFalse(r.matched)


# ============================================================================
# 5. SlangMapper — apply() method
# ============================================================================

class TestSlangMapperApply(unittest.TestCase):

    def setUp(self):
        self.sm = _make_mapper()

    def test_filler_dropped_by_default(self):
        out = self.sm.apply("dei nalaiku sollu")
        self.assertNotIn("dei", out)
        self.assertIn("tomorrow", out)
        self.assertIn("tell", out)

    def test_filler_kept_when_not_dropping(self):
        out = self.sm.apply("dei nalaiku", drop_fillers=False)
        # dei maps to "" even when not dropping (still empty string)
        # The filler sentinel is "<FILLER>" not the empty string —
        # display_normalized returns "". So it still disappears.
        self.assertIn("tomorrow", out)

    def test_multiple_tokens_glossed(self):
        out = self.sm.apply("nalaiku romba nalla")
        self.assertIn("tomorrow", out)
        self.assertIn("very", out)
        self.assertIn("good", out)

    def test_unknown_token_unchanged(self):
        out = self.sm.apply("meeting romba nalla irundhuchu")
        self.assertIn("meeting", out)       # unknown → passthrough
        self.assertIn("very", out)          # romba → very
        self.assertIn("good", out)          # nalla → good

    def test_negation_preserved(self):
        out = self.sm.apply("poga mateen")
        self.assertIn("go", out)            # poo → go
        self.assertIn("will not", out)      # mateen → will not

    def test_greeting_mapped(self):
        out = self.sm.apply("vanakkam nee eppadi irukka")
        self.assertIn("hello", out)
        self.assertIn("you", out)
        self.assertIn("how", out)

    def test_pronoun_mapped(self):
        out = self.sm.apply("naan pogiren")
        self.assertIn("I", out)
        self.assertIn("going", out)


# ============================================================================
# 6. SlangMapper — is_filler() and get_category()
# ============================================================================

class TestSlangMapperHelpers(unittest.TestCase):

    def setUp(self):
        self.sm = _make_mapper()

    def test_is_filler_true(self):
        self.assertTrue(self.sm.is_filler("dei"))
        self.assertTrue(self.sm.is_filler("machi"))
        self.assertTrue(self.sm.is_filler("da"))
        self.assertTrue(self.sm.is_filler("pa"))
        self.assertTrue(self.sm.is_filler("la"))

    def test_is_filler_false_for_content_words(self):
        self.assertFalse(self.sm.is_filler("sollu"))
        self.assertFalse(self.sm.is_filler("nalaiku"))
        self.assertFalse(self.sm.is_filler("vanakkam"))

    def test_is_filler_false_for_unknown(self):
        self.assertFalse(self.sm.is_filler("notaword"))

    def test_get_category_verb(self):
        self.assertEqual(self.sm.get_category("pannu"), "VERB")

    def test_get_category_temporal(self):
        self.assertEqual(self.sm.get_category("nalaiku"), "TEMPORAL")

    def test_get_category_filler(self):
        self.assertEqual(self.sm.get_category("dei"), "FILLER")

    def test_get_category_unknown(self):
        self.assertEqual(self.sm.get_category("unknown_word"), "UNKNOWN")


# ============================================================================
# 7. SlangMapper — entries_by_category() and entries_by_dialect()
# ============================================================================

class TestSlangMapperFilters(unittest.TestCase):

    def setUp(self):
        self.sm = _make_mapper()

    def test_entries_by_category_verb(self):
        verbs = self.sm.entries_by_category("VERB")
        self.assertGreater(len(verbs), 0)
        for e in verbs:
            self.assertEqual(e.category, "VERB")

    def test_entries_by_category_filler(self):
        fillers = self.sm.entries_by_category("FILLER")
        self.assertGreater(len(fillers), 0)
        for e in fillers:
            self.assertTrue(e.is_filler)

    def test_entries_by_category_temporal(self):
        temps = self.sm.entries_by_category("TEMPORAL")
        self.assertGreater(len(temps), 0)

    def test_entries_by_dialect_chennai(self):
        chennai = self.sm.entries_by_dialect("Chennai")
        self.assertGreater(len(chennai), 0)
        self.assertTrue(any(e.informal == "dei" for e in chennai))

    def test_entries_by_dialect_nellai(self):
        nellai = self.sm.entries_by_dialect("Nellai")
        self.assertTrue(any(e.informal in ("pa", "ppa") for e in nellai))

    def test_entries_by_dialect_standard(self):
        standard = self.sm.entries_by_dialect("Standard")
        self.assertGreater(len(standard), 10)


# ============================================================================
# 8. ExpressionMapper — CSV loading
# ============================================================================

class TestExpressionMapperLoading(unittest.TestCase):

    def setUp(self):
        self.em = _make_expr_mapper()

    def test_expr_csv_exists(self):
        self.assertTrue(_EXPR_CSV.exists(), f"Missing: {_EXPR_CSV}")

    def test_entries_loaded(self):
        self.assertGreater(len(self.em.all_entries()), 0)

    def test_vocabulary_non_empty(self):
        vocab = self.em.vocabulary()
        self.assertGreater(len(vocab), 0)

    def test_all_entries_have_normalized(self):
        for e in self.em.all_entries():
            self.assertIsInstance(e.normalized, str)
            self.assertGreater(len(e.normalized), 0)

    def test_all_entries_have_category(self):
        for e in self.em.all_entries():
            self.assertIsInstance(e.category, str)
            self.assertGreater(len(e.category), 0)

    def test_token_counts_correct(self):
        for e in self.em.all_entries():
            expected = len(e.expression.split())
            self.assertEqual(e.token_count, expected,
                f"Token count wrong for '{e.expression}'")


# ============================================================================
# 9. ExpressionMapper — Per-entry coverage (parametric over CSV)
# ============================================================================

class TestExpressionMapperAllEntries(unittest.TestCase):
    """Every row in personal_expressions.csv is tested."""

    @classmethod
    def setUpClass(cls):
        cls.em   = _make_expr_mapper()
        cls.rows = _load_csv_rows(_EXPR_CSV)

    def test_every_expression_in_vocabulary(self):
        missing = []
        vocab   = self.em.vocabulary()
        for row in self.rows:
            expr = row["expression"].strip().lower()
            if expr not in vocab:
                missing.append(expr)
        self.assertEqual(missing, [],
            f"Expressions not in vocabulary: {missing}")

    def test_every_expression_lookup_returns_entry(self):
        failures = []
        for row in self.rows:
            expr  = row["expression"].strip().lower()
            entry = self.em.lookup(expr)
            if entry is None:
                failures.append(expr)
        self.assertEqual(failures, [],
            f"lookup() returned None for: {failures}")

    def test_every_expression_normalized_correct(self):
        failures = []
        for row in self.rows:
            expr     = row["expression"].strip().lower()
            expected = row["normalized"].strip()
            entry    = self.em.lookup(expr)
            if entry and entry.normalized != expected:
                failures.append((expr, expected, entry.normalized))
        self.assertEqual(failures, [],
            f"Normalized mismatches: {failures[:5]}")

    def test_every_expression_maps_in_text(self):
        """
        Apply the expression as bare text to map() and confirm the
        normalized form appears in the output.
        """
        failures = []
        for row in self.rows:
            expr     = row["expression"].strip()
            expected = row["normalized"].strip().lower()
            result   = self.em.map(expr)
            if expected not in result.output_text.lower():
                failures.append((expr, expected, result.output_text))
        self.assertEqual(failures, [],
            f"map() did not produce expected output: {failures[:5]}")


# ============================================================================
# 10. ExpressionMapper — Specific known expressions
# ============================================================================

class TestExpressionMapperKnownExpressions(unittest.TestCase):

    def setUp(self):
        self.em = _make_expr_mapper()

    def _check_map(self, text: str, expected_in_output: str):
        result = self.em.map(text)
        self.assertIn(expected_in_output.lower(), result.output_text.lower(),
            f"map({text!r}) = {result.output_text!r}, expected '{expected_in_output}'")

    # Assistant commands
    def test_remind_pannu(self):    self._check_map("remind pannu",       "remind me")
    def test_remind_pannunga(self): self._check_map("remind pannunga",    "please remind me")
    def test_alarm_vai(self):       self._check_map("alarm vai",          "set alarm")
    def test_alarm_vaikku(self):    self._check_map("alarm vaikku",       "set alarm")
    def test_alarm_set_pannu(self): self._check_map("alarm set pannu",    "set alarm")
    def test_call_pannu(self):      self._check_map("call pannu",         "call")
    def test_call_pottu_kudu(self): self._check_map("call pottu kudu",    "call")
    def test_phone_pannu(self):     self._check_map("phone pannu",        "call")
    def test_phone_pottu_kudu(self):self._check_map("phone pottu kudu",   "call")
    def test_message_anuppu(self):  self._check_map("message anuppu",     "send message to")
    def test_message_podu(self):    self._check_map("message podu",       "send message to")
    def test_open_pannu(self):      self._check_map("open pannu",         "open")
    def test_close_pannu(self):     self._check_map("close pannu",        "close")
    def test_off_pannu(self):       self._check_map("off pannu",          "turn off")
    def test_on_pannu(self):        self._check_map("on pannu",           "turn on")
    def test_kammi_pannu(self):     self._check_map("kammi pannu",        "reduce")
    def test_koothu(self):          self._check_map("koothu",             "increase")
    def test_search_pannu(self):    self._check_map("search pannu",       "search")
    def test_vazhi_kaatu(self):     self._check_map("vazhi kaatu",        "directions to")
    def test_route_sollu(self):     self._check_map("route sollu",        "route to")

    # Query commands
    def test_paathu_sollu(self):    self._check_map("paathu sollu",       "check and tell me")
    def test_eppadi_iruku(self):    self._check_map("eppadi iruku",       "how is")
    def test_eppadi_iruka(self):    self._check_map("eppadi iruka",       "how are you")

    # Status statements
    def test_late_ah_varen(self):   self._check_map("late ah varen",      "I will come late")
    def test_aachi(self):           self._check_map("aachi",              "it's done")
    def test_okay_aachi(self):      self._check_map("okay aachi",         "it became okay")
    def test_problem_illai(self):   self._check_map("problem illai",      "no problem")
    def test_almost_aachi(self):    self._check_map("almost aachi",       "almost done")

    # Knowledge statements
    def test_theriyathu_da(self):   self._check_map("theriyathu da",      "don't know")
    def test_theriyum_da(self):     self._check_map("theriyum da",        "I know")

    # Activity commands
    def test_gym_poga(self):        self._check_map("gym poga",           "go to the gym")
    def test_submit_panna(self):    self._check_map("submit panna",       "to submit")

    # Weather
    def test_mazhai_varuma(self):   self._check_map("mazhai varuma",      "will it rain")

    # Greet / bye
    def test_vanakkam_solu(self):   self._check_map("vanakkam solu",      "greet")
    def test_bye_solu(self):        self._check_map("bye solu",           "say goodbye")


# ============================================================================
# 11. ExpressionMapper — Longest-match priority
# ============================================================================

class TestExpressionMapperLongestMatch(unittest.TestCase):
    """Longer expressions must win over shorter ones that share a prefix."""

    def setUp(self):
        self.em = _make_expr_mapper()

    def test_remind_panni_kudu_over_remind_pannu(self):
        """'remind panni kudu' (3 tokens) should win over 'remind pannu' (2)."""
        result = self.em.map("remind panni kudu")
        self.assertIn("please remind me", result.output_text.lower())
        # Crucially, 'remind me' from 'remind pannu' should NOT appear instead
        self.assertNotIn("remind me kudu", result.output_text.lower())

    def test_alarm_set_pannu_over_alarm_vai(self):
        """Each alarm expression should map correctly without interference."""
        r1 = self.em.map("alarm set pannu")
        self.assertIn("set alarm", r1.output_text.lower())
        r2 = self.em.map("alarm vai")
        self.assertIn("set alarm", r2.output_text.lower())

    def test_phone_pottu_kudu_over_phone_pannu(self):
        r = self.em.map("phone pottu kudu")
        self.assertIn("call", r.output_text.lower())
        # Must not partially match 'phone' and leave 'pottu kudu' behind
        self.assertNotIn("pottu kudu", r.output_text.lower())


# ============================================================================
# 12. ExpressionMapper — Semantic isolation (unrelated tokens unchanged)
# ============================================================================

class TestExpressionMapperSemanticIsolation(unittest.TestCase):
    """Matching an expression must not alter adjacent unrelated tokens."""

    def setUp(self):
        self.em = _make_expr_mapper()

    def test_adjacent_name_preserved(self):
        result = self.em.map("Priya ku message anuppu")
        self.assertIn("priya", result.output_text.lower())
        self.assertIn("send message to", result.output_text.lower())

    def test_adjacent_time_preserved(self):
        result = self.em.map("naalaikku remind pannu")
        self.assertIn("naalaikku", result.output_text.lower())
        self.assertIn("remind me", result.output_text.lower())

    def test_adjacent_app_preserved(self):
        result = self.em.map("whatsapp open pannu")
        self.assertIn("whatsapp", result.output_text.lower())
        self.assertIn("open", result.output_text.lower())

    def test_no_match_passes_through(self):
        result = self.em.map("completely unknown text here")
        self.assertEqual(result.n_matches, 0)
        self.assertIn("completely unknown text here", result.output_text.lower())

    def test_matches_recorded(self):
        result = self.em.map("remind pannu")
        self.assertGreater(result.n_matches, 0)
        self.assertEqual(result.matches[0].expression, "remind pannu")

    def test_to_dict_keys(self):
        result = self.em.map("open pannu")
        d = result.to_dict()
        for key in ("input", "output", "n_matches", "matches"):
            self.assertIn(key, d)


# ============================================================================
# 13. ExpressionMapper — entries_by_category()
# ============================================================================

class TestExpressionMapperCategories(unittest.TestCase):

    def setUp(self):
        self.em = _make_expr_mapper()

    def test_assistant_cmd_entries(self):
        cmds = self.em.entries_by_category("ASSISTANT_CMD")
        self.assertGreater(len(cmds), 0)
        for e in cmds:
            self.assertEqual(e.category, "ASSISTANT_CMD")

    def test_query_cmd_entries(self):
        entries = self.em.entries_by_category("QUERY_CMD")
        self.assertGreater(len(entries), 0)

    def test_status_stmt_entries(self):
        entries = self.em.entries_by_category("STATUS_STMT")
        self.assertGreater(len(entries), 0)

    def test_knowledge_stmt_entries(self):
        entries = self.em.entries_by_category("KNOWLEDGE_STMT")
        self.assertGreater(len(entries), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
