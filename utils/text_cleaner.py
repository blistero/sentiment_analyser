import re
import html
import unicodedata
from typing import Optional


_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002702-\U000027B0"
    "\U000024C2-\U0001F251"
    "]+",
    flags=re.UNICODE,
)

_URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML_PATTERN = re.compile(r"<[^>]+>")
_MENTION_PATTERN = re.compile(r"@\w+")
_HASHTAG_PATTERN = re.compile(r"#(\w+)")
_REPEATED_CHARS = re.compile(r"(.)\1{3,}")
_WHITESPACE = re.compile(r"\s+")
_NON_ASCII = re.compile(r"[^\x00-\x7F]+")


def clean_text(
    text: str,
    remove_urls: bool = True,
    remove_html: bool = True,
    remove_mentions: bool = True,
    expand_hashtags: bool = True,
    remove_emojis: bool = False,
    normalize_unicode: bool = True,
    fix_repeated_chars: bool = True,
    lowercase: bool = False,
    max_length: Optional[int] = None,
) -> str:
    if not text or not isinstance(text, str):
        return ""

    text = html.unescape(text)

    if normalize_unicode:
        text = unicodedata.normalize("NFKD", text)

    if remove_html:
        text = _HTML_PATTERN.sub(" ", text)

    if remove_urls:
        text = _URL_PATTERN.sub(" ", text)

    if remove_mentions:
        text = _MENTION_PATTERN.sub(" ", text)

    if expand_hashtags:
        text = _HASHTAG_PATTERN.sub(r"\1", text)

    if remove_emojis:
        text = _EMOJI_PATTERN.sub(" ", text)

    if fix_repeated_chars:
        # "sooooo good" -> "soo good" (keep 2 for emphasis signal)
        text = _REPEATED_CHARS.sub(r"\1\1", text)

    if lowercase:
        text = text.lower()

    text = _WHITESPACE.sub(" ", text).strip()

    if max_length:
        text = text[:max_length]

    return text


def normalize_rating_to_sentiment(rating: float) -> int:
    """Map star rating to sentiment label index (0=Neg, 1=Neutral, 2=Pos)."""
    if rating <= 2:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2


def truncate_text(text: str, max_words: int = 200) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def is_short_text(text: str, threshold: int = 10) -> bool:
    return len(text.split()) < threshold
