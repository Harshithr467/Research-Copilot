import re
from typing import Any, List, Mapping, Optional

from pydantic import BaseModel


class CitationOut(BaseModel):
    id: int
    doc: str
    page: Optional[int] = None
    author: Optional[str] = None
    year: Optional[int] = None
    formatted: Optional[str] = None


def _safe_author(result: Mapping[str, Any]) -> str:
    author = str(result.get("author") or "").strip()
    return author if author else "Unknown Author"


def _format_single_author_apa(author: str) -> str:
    author = author.strip()
    if not author:
        return ""
    if "et al" in author.lower():
        return author

    # Already looks like APA initials, e.g. "Soto, C. J."
    if "," in author and re.search(r"\b[A-Z]\.", author):
        return author

    parts = author.split()
    if len(parts) < 2:
        return author

    suffixes = {"Jr.", "Sr.", "II", "III", "IV"}
    suffix = ""
    if parts[-1] in suffixes:
        suffix = f", {parts.pop()}"

    last_name = parts[-1]
    initials = " ".join(f"{part[0]}." for part in parts[:-1] if part)
    return f"{last_name}, {initials}{suffix}".strip()


def _split_author_names(author: str) -> List[str]:
    normalized = re.sub(r"\s+(?:and|&)\s+", ", ", author.strip())
    names = [name.strip() for name in normalized.split(",") if name.strip()]

    # Preserve already-APA author strings such as "Soto, C. J., & John, O. P."
    if len(names) > 1 and any(re.fullmatch(r"(?:[A-Z]\.\s*)+", name) for name in names[1::2]):
        rebuilt: List[str] = []
        i = 0
        while i < len(names):
            if i + 1 < len(names) and re.fullmatch(r"(?:[A-Z]\.\s*)+", names[i + 1]):
                rebuilt.append(f"{names[i]}, {names[i + 1]}")
                i += 2
            else:
                rebuilt.append(names[i])
                i += 1
        return rebuilt

    return names


def format_authors_apa(author: str) -> str:
    authors = [_format_single_author_apa(name) for name in _split_author_names(author)]
    authors = [author for author in authors if author]

    if not authors:
        return "Unknown Author"
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{authors[0]}, & {authors[1]}"
    return f"{', '.join(authors[:-1])}, & {authors[-1]}"


def _format_single_author_mla_first(author: str) -> str:
    author = author.strip()
    if not author:
        return ""
    if "et al" in author.lower():
        return author

    # Already inverted, e.g. "Bhabha, Homi K."
    if "," in author:
        return author

    parts = author.split()
    if len(parts) < 2:
        return author

    suffixes = {"Jr.", "Sr.", "II", "III", "IV"}
    suffix = ""
    if parts[-1] in suffixes:
        suffix = f", {parts.pop()}"

    last_name = parts[-1]
    rest_of_name = " ".join(parts[:-1])
    return f"{last_name}, {rest_of_name}{suffix}".strip()


def _format_single_author_mla_following(author: str) -> str:
    author = author.strip()
    if not author:
        return ""

    # Convert simple APA initials back to a readable following author form.
    # Example: "John, O. P." -> "O. P. John"
    if "," in author:
        last_name, rest = [part.strip() for part in author.split(",", 1)]
        return f"{rest} {last_name}".strip()

    return author


def format_authors_mla(author: str) -> str:
    authors = [name for name in _split_author_names(author) if name]

    if not authors:
        return "Unknown Author"
    if len(authors) == 1:
        return _format_single_author_mla_first(authors[0])
    if len(authors) == 2:
        first = _format_single_author_mla_first(authors[0])
        second = _format_single_author_mla_following(authors[1])
        return f"{first}, and {second}"

    first = _format_single_author_mla_first(authors[0])
    following = [_format_single_author_mla_following(name) for name in authors[1:]]
    return f"{first}, {', '.join(following[:-1])}, and {following[-1]}"


def _safe_year(result: Mapping[str, Any]) -> str:
    year = result.get("year")
    return str(year) if isinstance(year, int) and year > 0 else "n.d."


def _safe_title(result: Mapping[str, Any]) -> str:
    title = str(result.get("title") or "").strip()
    return title if title else str(result.get("source_file") or "Uploaded document")


def _safe_source(result: Mapping[str, Any]) -> str:
    source = str(result.get("source_file") or "").strip()
    return source if source else "uploaded document"


def _page_number(result: Mapping[str, Any]) -> Optional[int]:
    page_number = result.get("page_number")
    if isinstance(page_number, int) and page_number > 0:
        return page_number
    return None


def _page_locator(result: Mapping[str, Any]) -> str:
    page_number = _page_number(result)
    if page_number is not None:
        return f"p. {page_number}"
    return f"chunk {result.get('chunk_id', 0)}"


def format_location(result: Mapping[str, Any], style: Optional[str]) -> str:
    """Format the retrieved source location in APA or MLA style."""
    author = _safe_author(result)
    year = _safe_year(result)
    title = _safe_title(result)
    source = _safe_source(result)
    locator = _page_locator(result)

    if style == "APA":
        return f"{format_authors_apa(author)}. ({year}). {title}. {source}, {locator}."
    if style == "MLA":
        mla_author = format_authors_mla(author)
        author_period = "" if mla_author.endswith(".") else "."
        return f'{mla_author}{author_period} "{title}." {source}, {year}, {locator}.'
    return f"{author}. {title}. {source}, {locator}."


def build_citation(result: Mapping[str, Any], citation_id: int, style: Optional[str]) -> CitationOut:
    year = result.get("year")
    citation_style = style or "APA"
    return CitationOut(
        id=citation_id,
        doc=_safe_source(result),
        page=_page_number(result),
        author=_safe_author(result),
        year=year if isinstance(year, int) and year > 0 else None,
        formatted=format_location(result, citation_style),
    )


def citation_key(result: Mapping[str, Any]) -> tuple[str, int | str]:
    source = _safe_source(result)
    page_number = _page_number(result)
    if page_number is not None:
        return (source, page_number)
    return (source, f"chunk:{result.get('chunk_id', 0)}")
