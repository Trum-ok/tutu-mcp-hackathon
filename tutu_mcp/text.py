"""Text matching shared by the two checks that read the user's own words.

The premise gate and the groundedness check both ask the same question — does
this exact value appear in what the user actually wrote — and both documented
themselves as using "the same rule as the other". Two identical copies made that
a promise rather than a fact: a fix to one silently split the behaviour the docs
claim is shared.
"""

import re
from functools import lru_cache


@lru_cache(maxsize=512)
def word_bounded(value: str) -> re.Pattern[str]:
    """`value` as a whole word: «Мск» must not match inside «Мскво».

    Word boundaries are spelled with lookarounds rather than `\\b` because the
    values are Russian text and often end in punctuation, where `\\b` flips
    meaning depending on the last character.
    """
    return re.compile(rf"(?<!\w){re.escape(value)}(?!\w)", re.IGNORECASE)
