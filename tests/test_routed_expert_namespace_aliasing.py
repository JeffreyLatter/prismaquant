"""A packed-expert Linear has three spellings; the classifier must not read
the difference between them as a conflicting declaration.

`ProfileRoutedExpertClassifier.classify` asks the profile for a format group
under each of the live, recipe and vLLM-internal names, then insists the
answers agree. `packed_expert_format_group` builds its key as
``f"{parent}::__packed_format__:{projections}"`` from whichever spelling it was
handed, so on any model whose namespaces differ by a prefix the raw keys can
never agree. A multimodal MoE wrapper is exactly that shape: the checkpoint
carries ``model.language_model.layers.N...`` and vLLM ``language_model.model.
layers.N...`` against a bare ``model.layers.N...`` recipe name.

Ornith-1.5-35B-A3B hit this on the AURA cost stage:

    RuntimeError: profile Qwen3_5Profile assigns conflicting routed-expert
    groups to 'model.layers.0.mlp.experts.gate_up_proj':
    ['language_model.model.layers.0.mlp.experts::__packed_format__:...',
     'model.layers.0.mlp.experts::__packed_format__:...']

Both keys describe one physical group. What a real conflict looks like -- two
different projection groupings, or two spellings pointing at different layers
-- must still refuse.
"""

import pytest

from prismaquant.routed_experts import ProfileRoutedExpertClassifier

PROJECTIONS = ("gate_up_proj", "down_proj")
GROUPING = "__packed_format__:gate_up_proj,down_proj"


class _MultimodalMoEProfile:
    """Minimal profile whose three namespaces differ by prefix only."""

    def packed_expert_param_names(self):
        return PROJECTIONS

    def unpacked_expert_projection_names(self):
        return ()

    def packed_expert_projection_names(self, packed_name):
        return PROJECTIONS

    def vllm_fused_moe_scheme_projection_names(self, packed_name):
        return PROJECTIONS

    def per_expert_moe_regex(self):
        return None

    def live_to_recipe_name(self, qname):
        return qname.replace("model.language_model.", "model.")

    def to_vllm_internal_name(self, recipe):
        return recipe.replace("model.", "language_model.model.", 1)

    def packed_expert_format_group(self, qname):
        parent, _, leaf = qname.rpartition(".")
        if leaf not in PROJECTIONS:
            return None
        return f"{parent}::{GROUPING}"


class _LayerConfusedProfile(_MultimodalMoEProfile):
    """A profile whose spellings disagree about WHICH layer -- a real bug."""

    def to_vllm_internal_name(self, recipe):
        return recipe.replace("layers.0.", "layers.7.")


class _GroupingConfusedProfile(_MultimodalMoEProfile):
    """A profile that groups one Linear two different ways -- a real bug."""

    def packed_expert_format_group(self, qname):
        parent, _, leaf = qname.rpartition(".")
        if leaf not in PROJECTIONS:
            return None
        if parent.startswith("language_model."):
            return f"{parent}::__packed_format__:gate_up_proj"
        return f"{parent}::{GROUPING}"


LIVE = "model.language_model.layers.0.mlp.experts.gate_up_proj"


def test_prefix_only_namespace_difference_is_not_a_conflict():
    match = ProfileRoutedExpertClassifier(_MultimodalMoEProfile()).classify(LIVE)
    assert match is not None
    assert match.projection_name == "gate_up_proj"
    assert match.group_key.endswith(GROUPING)


def test_group_key_is_deterministic_and_fuses_siblings():
    """Siblings must land on ONE key or union-find will not fuse them."""
    classifier = ProfileRoutedExpertClassifier(_MultimodalMoEProfile())
    gate_up = classifier.classify(LIVE)
    down = classifier.classify(LIVE.replace("gate_up_proj", "down_proj"))
    assert gate_up.group_key == down.group_key
    # and stable across constructions -- the old code picked an arbitrary
    # element out of a set
    again = ProfileRoutedExpertClassifier(_MultimodalMoEProfile()).classify(LIVE)
    assert again.group_key == gate_up.group_key


def test_disagreeing_layer_position_still_refuses():
    with pytest.raises(RuntimeError, match="conflicting routed-expert groups"):
        ProfileRoutedExpertClassifier(_LayerConfusedProfile()).classify(LIVE)


def test_disagreeing_projection_grouping_still_refuses():
    with pytest.raises(RuntimeError, match="conflicting routed-expert groups"):
        ProfileRoutedExpertClassifier(_GroupingConfusedProfile()).classify(LIVE)
