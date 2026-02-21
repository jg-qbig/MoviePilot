from src.lib.utils import prompt_gemini, results_to_str


def generate(question: str, results: list[dict]) -> str:
    prompt = f"""
    Answer the question below based on the provided movie descriptions.

    Question: {question}

    Descriptions: {results_to_str(results)}

    Answer:
    """

    return prompt_gemini(prompt)


def summarize(question: str, results: list[dict]) -> str:
    prompt = f"""
    You will be provided with a list movies and their descriptions as context.
    Answer the question below by generating a concise and coherent summary from the given Context.
    Condense the context into a well-written summary that captures the main ideas, key points, and insights presented in the context.
    Prioritize clarity and brevity while retaining the essential information.
    Aim to convey the context's core message and any supporting details that contribute to a comprehensive understanding.
    Craft the summary to be self-contained, ensuring that readers can grasp the content even if they haven't read the context.
    Provide context where necessary and avoid excessive technical jargon or verbosity.
    The goal is to create a summary that effectively communicates the context's content while being easily digestible and engaging.
    Provide a comprehensive answer in a single paragraph.

    Question: {question}

    Context: {results_to_str(results)}

    Summary:
    """
    return prompt_gemini(prompt)


def summarize_with_citations(question: str, results: list[dict]) -> str:
    prompt = f"""
    You will be provided with a list movies and their descriptions as context.
    Answer the question below by generating a concise and coherent summary from the given Context.
    Condense the context into a well-written summary that captures the main ideas, key points, and insights presented in the context.
    Prioritize clarity and brevity while retaining the essential information.
    Aim to convey the context's core message and any supporting details that contribute to a comprehensive understanding.
    Craft the summary to be self-contained, ensuring that readers can grasp the content even if they haven't read the context.
    Provide context where necessary and avoid excessive technical jargon or verbosity.
    The goal is to create a summary that effectively communicates the context's content while being easily digestible and engaging.
    To complete your summary you must cite all information extracted from the movie descriptions by referencing the number in front of each movie title.

    Question: {question}

    Context: {results_to_str(results)}

    Answer:
    """
    print(prompt)
    return prompt_gemini(prompt)


def question_answering(question: str, results: list[dict]) -> str:
    prompt = f"""
    You are an assistant for question-answering tasks. Use the following descriptions of retrieved movies to answer the question. 

    If the context provides information to answer the question, respond with a concise answer in three sentences maximum using that information.

    If the context does not provide information, simply say that you don't know.

    Question: {question}

    Context: {results_to_str(results)}

    Answer:
    """
    return prompt_gemini(prompt)
