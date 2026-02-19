from src.lib.query_enhancement import prompt_gemini


def augment_prompt(query: str, docs: list):
    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Query: {query}

    Documents:
    {docs}

    Provide a comprehensive answer that addresses the query:"""

    return prompt_gemini(prompt)


def summarize(query: str, results: list):
    prompt = f"""
    Provide information useful to this query by synthesizing information from multiple search results in detail.
    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
    This should be tailored to Hoopla users. Hoopla is a movie streaming service.
    Query: {query}
    Search Results:
    {results}
    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
    """

    return prompt_gemini(prompt)


def summarize_with_citations(query: str, results: list):
    prompt = f"""Answer the question or provide information based on the provided documents.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

    Query: {query}

    Documents:
    {results}

    Instructions:
    - Provide a comprehensive answer that addresses the query
    - Cite sources using [1], [2], etc. format when referencing information
    - If sources disagree, mention the different viewpoints
    - If the answer isn't in the documents, say "I don't have enough information"
    - Be direct and informative

    Answer:"""

    return prompt_gemini(prompt)


def question_answering(question: str, results: list):
    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Question: {question}

    Documents:
    {results}

    Instructions:
    - Answer questions directly and concisely
    - Be casual and conversational
    - Don't be cringe or hype-y
    - Talk like a normal person would in a chat conversation

    Answer:"""

    return prompt_gemini(prompt)
