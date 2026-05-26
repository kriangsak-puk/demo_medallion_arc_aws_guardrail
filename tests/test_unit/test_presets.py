"""Unit tests for the PresetManager and AttackPreset."""

import re

import pytest

from aws_demo_booth.presets import AttackPreset, PresetManager

# Regex pattern to match common emoji characters (Unicode emoji ranges)
EMOJI_PATTERN = re.compile(
    "[\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map symbols
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002702-\U000027B0"  # dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended-A
    "\U00002600-\U000026FF"  # misc symbols
    "\U0000FE00-\U0000FE0F"  # variation selectors
    "\U0000200D"  # zero width joiner
    "]"
)


@pytest.fixture
def manager() -> PresetManager:
    return PresetManager()


class TestAttackPreset:
    """Tests for the AttackPreset dataclass."""

    def test_create_attack_preset(self):
        preset = AttackPreset(
            id="test_id",
            label="🔓 Test Label",
            emoji="🔓",
            category="pii_extraction",
            prompt_text="Test prompt text",
        )
        assert preset.id == "test_id"
        assert preset.label == "🔓 Test Label"
        assert preset.emoji == "🔓"
        assert preset.category == "pii_extraction"
        assert preset.prompt_text == "Test prompt text"
        assert preset.is_analytics is False

    def test_create_analytics_preset(self):
        preset = AttackPreset(
            id="analytics_test",
            label="📊 Test Analytics",
            emoji="📊",
            category="analytics",
            prompt_text="Show me data",
            is_analytics=True,
        )
        assert preset.is_analytics is True


class TestPresetManager:
    """Tests for the PresetManager class."""

    def test_get_all_presets_returns_list(self, manager: PresetManager):
        presets = manager.get_all_presets()
        assert isinstance(presets, list)
        assert len(presets) > 0

    def test_total_preset_count(self, manager: PresetManager):
        all_presets = manager.get_all_presets()
        attack = manager.get_attack_presets()
        analytics = manager.get_analytics_presets()
        assert len(all_presets) == len(attack) + len(analytics)

    def test_attack_presets_count_between_5_and_10(self, manager: PresetManager):
        attack = manager.get_attack_presets()
        assert 5 <= len(attack) <= 10

    def test_analytics_presets_count_between_2_and_4(self, manager: PresetManager):
        analytics = manager.get_analytics_presets()
        assert 2 <= len(analytics) <= 4

    def test_attack_presets_not_analytics(self, manager: PresetManager):
        for preset in manager.get_attack_presets():
            assert preset.is_analytics is False

    def test_analytics_presets_are_analytics(self, manager: PresetManager):
        for preset in manager.get_analytics_presets():
            assert preset.is_analytics is True

    def test_all_labels_max_30_chars(self, manager: PresetManager):
        for preset in manager.get_all_presets():
            assert len(preset.label) <= 30, f"Label too long: '{preset.label}' ({len(preset.label)} chars)"

    def test_all_labels_contain_emoji(self, manager: PresetManager):
        for preset in manager.get_all_presets():
            has_emoji = bool(EMOJI_PATTERN.search(preset.label))
            assert has_emoji, f"Label missing emoji: '{preset.label}'"

    def test_all_presets_have_unique_ids(self, manager: PresetManager):
        ids = [p.id for p in manager.get_all_presets()]
        assert len(ids) == len(set(ids)), "Duplicate preset IDs found"

    def test_get_preset_by_id_found(self, manager: PresetManager):
        all_presets = manager.get_all_presets()
        first = all_presets[0]
        result = manager.get_preset_by_id(first.id)
        assert result is not None
        assert result.id == first.id

    def test_get_preset_by_id_not_found(self, manager: PresetManager):
        result = manager.get_preset_by_id("nonexistent_id")
        assert result is None

    def test_attack_categories_cover_required_types(self, manager: PresetManager):
        categories = {p.category for p in manager.get_attack_presets()}
        required = {"pii_extraction", "prompt_injection", "role_escalation", "data_exfiltration", "hallucination_induction"}
        assert required.issubset(categories), f"Missing categories: {required - categories}"

    def test_all_presets_have_nonempty_prompt_text(self, manager: PresetManager):
        for preset in manager.get_all_presets():
            assert len(preset.prompt_text.strip()) > 0

    def test_get_all_presets_returns_copy(self, manager: PresetManager):
        """Ensure get_all_presets returns a new list (not a reference to internal state)."""
        presets1 = manager.get_all_presets()
        presets2 = manager.get_all_presets()
        assert presets1 is not presets2
