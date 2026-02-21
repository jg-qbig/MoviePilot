from src.lib.utils import prompt_gemini


def correct_spelling(query: str) -> str:
    prompt = f"""Fix any spelling errors in this movie search query.
    Only correct obvious typos. Do not change correctly spelled words.

    Query: "{query}"

    If there are no errors, return the original query.

    Corrected:"""

    return prompt_gemini(prompt)


def rewrite_query(query: str) -> str:
    prompt = f"""Rewrite this movie search query to be more specific and searchable.

    Original: "{query}"

    Consider:
    - Common movie knowledge (famous actors, popular films)
    - Genre conventions (horror = scary, animation = cartoon)
    - Keep it concise (under 10 words)
    - It should be a google style search query that's very specific
    - Don't use boolean logic

    Examples:
    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

    Rewritten query:"""

    return prompt_gemini(prompt)


def expand_query(query: str) -> str:
    prompt = f"""Expand this movie search query with related terms.

    Add synonyms and related concepts that might appear in movie descriptions.
    Keep expansions relevant and focused.
    This will be appended to the original query.

    Examples:
    - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
    - "action movie with bear" -> "action thriller bear chase fight adventure"
    - "comedy with bear" -> "comedy funny bear humor lighthearted"

    Query: "{query}"
    """

    return prompt_gemini(prompt)


def enhance_query(query: str, method: str = "") -> str:
    if method == "spell":
        enhanced_query = correct_spelling(query)
        print(f"Enhanced query (spell): '{query}' -> '{enhanced_query}'\n")
        return enhanced_query
    if method == "rewrite":
        enhanced_query = rewrite_query(query)
        print(f"Enhanced query (rewrite): '{query}' -> '{enhanced_query}'\n")
        return enhanced_query
    if method == "expand":
        enhanced_query = expand_query(query)
        print(f"Enhanced query (expand): '{query}' -> '{enhanced_query}'\n")
        return enhanced_query
    return query
