from src.lib.utils import prompt_gemini


def correct_spelling(query: str) -> str:
    prompt = f"""
    Act as a spelling corrector and improver. Reply to each message only with the rewritten text
    Strictly follow these rules:
    - Correct spelling and grammar
    - ALWAYS detect and maintain the original language of the text
    - NEVER surround the rewritten text with quotes
    - Do not replace urls with markdown links
    - Do not change emojis

    Text to rewrite: {query}

    Rewritten text:
    """
    return prompt_gemini(prompt)


def rewrite_query(query: str) -> str:
    prompt = f"""
    You are tasked with reqriting the below movie search query to be more specific and optimized for vectorstore retrieval.
    to complete your task, consider common movie knowledge (famous actors, popular films) and genre conventions (horror = scary, animation = cartoon).
    Keep the query concise and specific. Avoid boolean logic.

    Examples:
    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

    Query: {query}

    Rewritten query:"""

    return prompt_gemini(prompt)


def expand_query(query: str) -> str:
    prompt = f"""
    You are an AI language model assistant.
    Your task is to perform query expansion on the movie search query below by extending the query with additional search terms to retrieve relevant documents from a vector database.
    By generating multiple perspectives on the user question, your goal is to help the user do an adequate covering of the distance-based similarity search.
    Think in pictures meaning that your questions should cover the largest possible perspective.

    Examples:
    - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
    - "action movie with bear" -> "action thriller bear chase fight adventure"
    - "comedy with bear" -> "comedy funny bear humor lighthearted"

    Query: "{query}"

    Expanded Query:
    """
    return prompt_gemini(prompt)


# Prompt inspirations from https://smith.langchain.com/hub/search?q=evaluation


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
