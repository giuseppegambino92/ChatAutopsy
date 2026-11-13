"""Le regex lessicali a trie devono dare esattamente gli stessi match dell'alternanza piatta di riferimento."""
from __future__ import annotations

import pytest

from chatautopsy import config, demo, ingestion

LEXICONS = ["CONFLICT_LEXICON", "ACCUSATION_LEXICON", "NEGATION_LEXICON", "AFFECTION_LEXICON", "APOLOGY_LEXICON",
            "BLAME_LEXICON", "GASLIGHTING_LEXICON", "PASSIVE_AGGRESSIVE_LEXICON", "GUILT_TRIPPING_LEXICON",
            "WE_PRONOUNS", "I_PRONOUNS", "YOU_PRONOUNS", "ABSOLUTE_TERMS", "PET_NAME_TERMS"]
EDGE = ["mi dispiace tanto davvero", "fine. ok", "va bene.", "scusa scusami", "tu sempre  sempre", ":( no",
        "amore mio amoremio", "SEI SEMPRE LA SOLITA!!", "è colpa tua", "non è vero, te lo sei inventato"]


@pytest.mark.parametrize("name", LEXICONS)
def test_trie_matches_flat_reference(name):
    terms = getattr(config, name)
    fast, flat = ingestion.compile_lexicon(terms), ingestion._compile_lexicon_flat(terms)
    corpus = demo.couple_chat(days=120).lower().splitlines() + EDGE
    for line in corpus:
        assert fast.findall(line) == flat.findall(line), (name, line)
