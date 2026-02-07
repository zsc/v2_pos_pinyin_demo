"""Tests for the rules engine."""

from __future__ import annotations

import unittest

from pinyinize.rules import AppliedRule, _match_part, rule_matches, sort_rules
from pinyinize.types import Token


class TestMatchPart(unittest.TestCase):
    """Tests for _match_part function."""

    def test_match_text_exact(self) -> None:
        """Test exact text match."""
        part = {"text": "银行"}
        tok = Token(span_id="S0", index_in_span=0, start=0, end=2, text="银行")
        self.assertTrue(_match_part(part, tok))

    def test_match_text_mismatch(self) -> None:
        """Test text mismatch."""
        part = {"text": "银行"}
        tok = Token(span_id="S0", index_in_span=0, start=0, end=2, text="学校")
        self.assertFalse(_match_part(part, tok))

    def test_match_text_in(self) -> None:
        """Test text_in match."""
        part = {"text_in": ["银行", "学校", "医院"]}
        tok = Token(span_id="S0", index_in_span=0, start=0, end=2, text="学校")
        self.assertTrue(_match_part(part, tok))

    def test_match_text_in_mismatch(self) -> None:
        """Test text_in mismatch."""
        part = {"text_in": ["银行", "学校"]}
        tok = Token(span_id="S0", index_in_span=0, start=0, end=2, text="医院")
        self.assertFalse(_match_part(part, tok))

    def test_match_regex(self) -> None:
        """Test regex match."""
        part = {"regex": r"^银.+"}
        tok = Token(span_id="S0", index_in_span=0, start=0, end=2, text="银行")
        self.assertTrue(_match_part(part, tok))

    def test_match_regex_mismatch(self) -> None:
        """Test regex mismatch."""
        part = {"regex": r"^银.+"}
        tok = Token(span_id="S0", index_in_span=0, start=0, end=2, text="学校")
        self.assertFalse(_match_part(part, tok))

    def test_match_upos_in(self) -> None:
        """Test upos_in match."""
        part = {"upos_in": ["NOUN", "PROPN"]}
        tok = Token(
            span_id="S0", index_in_span=0, start=0, end=2, text="银行",
            upos="NOUN", xpos="NN", ner="O"
        )
        self.assertTrue(_match_part(part, tok))

    def test_match_upos_in_mismatch(self) -> None:
        """Test upos_in mismatch."""
        part = {"upos_in": ["VERB", "ADJ"]}
        tok = Token(
            span_id="S0", index_in_span=0, start=0, end=2, text="银行",
            upos="NOUN", xpos="NN", ner="O"
        )
        self.assertFalse(_match_part(part, tok))

    def test_match_xpos_in(self) -> None:
        """Test xpos_in match."""
        part = {"xpos_in": ["NN", "NR"]}
        tok = Token(
            span_id="S0", index_in_span=0, start=0, end=2, text="银行",
            upos="NOUN", xpos="NN", ner="O"
        )
        self.assertTrue(_match_part(part, tok))

    def test_match_ner_in(self) -> None:
        """Test ner_in match."""
        part = {"ner_in": ["ORG", "PER"]}
        tok = Token(
            span_id="S0", index_in_span=0, start=0, end=2, text="银行",
            upos="NOUN", xpos="NN", ner="ORG"
        )
        self.assertTrue(_match_part(part, tok))

    def test_match_contains(self) -> None:
        """Test contains match - all specified chars must be present."""
        # "银" is in "银行", but "金" is not, so this fails
        part = {"contains": ["银", "金"]}
        tok = Token(span_id="S0", index_in_span=0, start=0, end=2, text="银行")
        self.assertFalse(_match_part(part, tok))

        # Just "银" is in "银行"
        part = {"contains": ["银"]}
        self.assertTrue(_match_part(part, tok))

    def test_match_contains_mismatch(self) -> None:
        """Test contains mismatch."""
        part = {"contains": ["金", "铜"]}
        tok = Token(span_id="S0", index_in_span=0, start=0, end=2, text="银行")
        self.assertFalse(_match_part(part, tok))

    def test_match_multiple_criteria(self) -> None:
        """Test matching multiple criteria."""
        part = {"text": "银行", "upos_in": ["NOUN"], "ner_in": ["ORG", "O"]}
        tok = Token(
            span_id="S0", index_in_span=0, start=0, end=2, text="银行",
            upos="NOUN", xpos="NN", ner="O"
        )
        self.assertTrue(_match_part(part, tok))

    def test_match_multiple_criteria_fail(self) -> None:
        """Test multiple criteria where one fails."""
        part = {"text": "银行", "upos_in": ["VERB"]}
        tok = Token(
            span_id="S0", index_in_span=0, start=0, end=2, text="银行",
            upos="NOUN", xpos="NN", ner="O"
        )
        self.assertFalse(_match_part(part, tok))

    def test_empty_part(self) -> None:
        """Test empty part matches anything."""
        part = {}
        tok = Token(span_id="S0", index_in_span=0, start=0, end=2, text="银行")
        self.assertTrue(_match_part(part, tok))


class TestRuleMatches(unittest.TestCase):
    """Tests for rule_matches function."""

    def test_match_self_only(self) -> None:
        """Test rule with only self match."""
        rule = {
            "id": "test_rule",
            "priority": 100,
            "match": {"self": {"text": "银行"}},
            "target": {"char": "行", "occurrence": 1},
            "choose": "háng",
        }
        tok = Token(span_id="S0", index_in_span=0, start=0, end=2, text="银行")
        self.assertTrue(rule_matches(rule, tok, None, None))

    def test_match_self_fail(self) -> None:
        """Test rule where self doesn't match."""
        rule = {
            "id": "test_rule",
            "match": {"self": {"text": "学校"}},
            "target": {"char": "校", "occurrence": 1},
            "choose": "xiào",
        }
        tok = Token(span_id="S0", index_in_span=0, start=0, end=2, text="银行")
        self.assertFalse(rule_matches(rule, tok, None, None))

    def test_match_with_prev(self) -> None:
        """Test rule matching previous token."""
        rule = {
            "id": "test_rule",
            "match": {
                "self": {"text": "银行"},
                "prev": {"text": "中国"},
            },
            "target": {"char": "行", "occurrence": 1},
            "choose": "háng",
        }
        prev_tok = Token(span_id="S0", index_in_span=0, start=0, end=2, text="中国")
        tok = Token(span_id="S0", index_in_span=1, start=2, end=4, text="银行")
        self.assertTrue(rule_matches(rule, tok, prev_tok, None))

    def test_match_with_prev_fail_no_prev(self) -> None:
        """Test rule with prev match but no previous token."""
        rule = {
            "id": "test_rule",
            "match": {
                "self": {"text": "银行"},
                "prev": {"text": "中国"},
            },
            "target": {"char": "行", "occurrence": 1},
            "choose": "háng",
        }
        tok = Token(span_id="S0", index_in_span=0, start=0, end=2, text="银行")
        self.assertFalse(rule_matches(rule, tok, None, None))

    def test_match_with_next(self) -> None:
        """Test rule matching next token."""
        rule = {
            "id": "test_rule",
            "match": {
                "self": {"text": "中国"},
                "next": {"text": "银行"},
            },
            "target": {"char": "行", "occurrence": 1},
            "choose": "háng",
        }
        tok = Token(span_id="S0", index_in_span=0, start=0, end=2, text="中国")
        next_tok = Token(span_id="S0", index_in_span=1, start=2, end=4, text="银行")
        self.assertTrue(rule_matches(rule, tok, None, next_tok))

    def test_match_with_next_fail_no_next(self) -> None:
        """Test rule with next match but no next token."""
        rule = {
            "id": "test_rule",
            "match": {
                "self": {"text": "中国"},
                "next": {"text": "银行"},
            },
            "target": {"char": "行", "occurrence": 1},
            "choose": "háng",
        }
        tok = Token(span_id="S0", index_in_span=0, start=0, end=2, text="中国")
        self.assertFalse(rule_matches(rule, tok, None, None))

    def test_match_all_three(self) -> None:
        """Test rule matching self, prev, and next."""
        rule = {
            "id": "test_rule",
            "match": {
                "prev": {"text": "中国"},
                "self": {"text": "的"},
                "next": {"text": "银行"},
            },
            "target": {"char": "的", "occurrence": 1},
            "choose": "de",
        }
        prev_tok = Token(span_id="S0", index_in_span=0, start=0, end=2, text="中国")
        tok = Token(span_id="S0", index_in_span=1, start=2, end=3, text="的")
        next_tok = Token(span_id="S0", index_in_span=2, start=3, end=5, text="银行")
        self.assertTrue(rule_matches(rule, tok, prev_tok, next_tok))

    def test_invalid_match_type(self) -> None:
        """Test rule with invalid match type."""
        rule = {
            "id": "test_rule",
            "match": "invalid",  # type: ignore[dict-item]
            "target": {"char": "行", "occurrence": 1},
            "choose": "háng",
        }
        tok = Token(span_id="S0", index_in_span=0, start=0, end=2, text="银行")
        self.assertFalse(rule_matches(rule, tok, None, None))

    def test_invalid_self_part(self) -> None:
        """Test rule with invalid self part."""
        rule = {
            "id": "test_rule",
            "match": {"self": "invalid"},  # type: ignore[dict-item]
            "target": {"char": "行", "occurrence": 1},
            "choose": "háng",
        }
        tok = Token(span_id="S0", index_in_span=0, start=0, end=2, text="银行")
        self.assertFalse(rule_matches(rule, tok, None, None))


class TestSortRules(unittest.TestCase):
    """Tests for sort_rules function."""

    def test_sort_by_priority_desc(self) -> None:
        """Test sorting by priority descending."""
        rules = [
            {"id": "low", "priority": 10},
            {"id": "high", "priority": 100},
            {"id": "medium", "priority": 50},
        ]
        sorted_rules = sort_rules(rules)  # type: ignore[arg-type]
        self.assertEqual([r["id"] for r in sorted_rules], ["high", "medium", "low"])

    def test_sort_by_id_asc_when_same_priority(self) -> None:
        """Test sorting by ID ascending when priority is equal."""
        rules = [
            {"id": "z", "priority": 50},
            {"id": "a", "priority": 50},
            {"id": "m", "priority": 50},
        ]
        sorted_rules = sort_rules(rules)  # type: ignore[arg-type]
        self.assertEqual([r["id"] for r in sorted_rules], ["a", "m", "z"])

    def test_sort_mixed(self) -> None:
        """Test sorting with mixed priorities."""
        rules = [
            {"id": "a_low", "priority": 10},
            {"id": "b_high", "priority": 100},
            {"id": "c_low", "priority": 10},
            {"id": "d_medium", "priority": 50},
        ]
        sorted_rules = sort_rules(rules)  # type: ignore[arg-type]
        ids = [r["id"] for r in sorted_rules]
        self.assertEqual(ids[0], "b_high")
        self.assertEqual(ids[1], "d_medium")
        # a_low and c_low should come after, in alphabetical order
        self.assertIn(ids[2], ["a_low", "c_low"])
        self.assertIn(ids[3], ["a_low", "c_low"])

    def test_sort_missing_priority(self) -> None:
        """Test sorting with missing priority defaults to 0."""
        rules = [
            {"id": "explicit", "priority": 10},
            {"id": "missing"},  # No priority
        ]
        sorted_rules = sort_rules(rules)  # type: ignore[arg-type]
        self.assertEqual([r["id"] for r in sorted_rules], ["explicit", "missing"])

    def test_sort_empty(self) -> None:
        """Test sorting empty list."""
        self.assertEqual(sort_rules([]), [])


class TestAppliedRule(unittest.TestCase):
    """Tests for AppliedRule dataclass."""

    def test_creation(self) -> None:
        """Test AppliedRule creation."""
        rule = AppliedRule(
            rule_id="test_001",
            token_start=0,
            token_end=2,
            token_text="银行",
            target_char="行",
            occurrence=1,
            choose="háng",
        )
        self.assertEqual(rule.rule_id, "test_001")
        self.assertEqual(rule.token_start, 0)
        self.assertEqual(rule.token_end, 2)
        self.assertEqual(rule.token_text, "银行")
        self.assertEqual(rule.target_char, "行")
        self.assertEqual(rule.occurrence, 1)
        self.assertEqual(rule.choose, "háng")

    def test_with_occurrence_all(self) -> None:
        """Test AppliedRule with occurrence='all'."""
        rule = AppliedRule(
            rule_id="test_002",
            token_start=0,
            token_end=3,
            token_text="行行好",
            target_char="行",
            occurrence="all",
            choose="xíng",
        )
        self.assertEqual(rule.occurrence, "all")


class TestComplexRuleScenarios(unittest.TestCase):
    """Complex real-world rule scenarios."""

    def test_bank_scenario(self) -> None:
        """Test rule for '银行' in context."""
        rule = {
            "id": "bank_rule",
            "priority": 100,
            "match": {
                "self": {"text_in": ["银行", "支行", "本行"]},
                "prev": {"upos_in": ["NOUN", "PROPN"]},
            },
            "target": {"char": "行", "occurrence": 1},
            "choose": "háng",
        }
        prev_tok = Token(
            span_id="S0", index_in_span=0, start=0, end=1, text="中",
            upos="NOUN", xpos="NN", ner="O"
        )
        tok = Token(
            span_id="S0", index_in_span=1, start=1, end=3, text="银行",
            upos="NOUN", xpos="NN", ner="O"
        )
        self.assertTrue(rule_matches(rule, tok, prev_tok, None))

    def test_walk_scenario(self) -> None:
        """Test rule for '行走' context."""
        rule = {
            "id": "walk_rule",
            "priority": 100,
            "match": {
                "self": {"upos_in": ["VERB"]},
                "contains": ["行"],
            },
            "target": {"char": "行", "occurrence": 1},
            "choose": "xíng",
        }
        tok = Token(
            span_id="S0", index_in_span=0, start=0, end=2, text="行走",
            upos="VERB", xpos="VV", ner="O"
        )
        self.assertTrue(rule_matches(rule, tok, None, None))

    def test_grow_scenario(self) -> None:
        """Test rule for '长大' context."""
        rule = {
            "id": "grow_rule",
            "priority": 100,
            "match": {
                "self": {"text_in": ["长大", "长高", "长壮"]},
            },
            "target": {"char": "长", "occurrence": 1},
            "choose": "zhǎng",
        }
        tok = Token(
            span_id="S0", index_in_span=0, start=0, end=2, text="长大",
            upos="VERB", xpos="VV", ner="O"
        )
        self.assertTrue(rule_matches(rule, tok, None, None))

    def test_long_scenario(self) -> None:
        """Test rule for '很长' context."""
        rule = {
            "id": "long_rule",
            "priority": 100,
            "match": {
                "self": {"upos_in": ["ADJ"]},
                "prev": {"text_in": ["很", "太", "非常"]},
            },
            "target": {"char": "长", "occurrence": 1},
            "choose": "cháng",
        }
        prev_tok = Token(
            span_id="S0", index_in_span=0, start=0, end=1, text="很",
            upos="ADV", xpos="AD", ner="O"
        )
        tok = Token(
            span_id="S0", index_in_span=1, start=1, end=2, text="长",
            upos="ADJ", xpos="JJ", ner="O"
        )
        self.assertTrue(rule_matches(rule, tok, prev_tok, None))


if __name__ == "__main__":
    unittest.main()
