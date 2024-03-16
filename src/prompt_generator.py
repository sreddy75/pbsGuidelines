from langchain.prompts import PromptTemplate

class PromptGenerator:

    template = """
        Below is an query that may be asked.
        Your goal is to:
        - Search all of the document for the query
        - provide an detailed explanation

        QUERY: {query}
        YOUR RESPONSE:
    """

    prompt = PromptTemplate(
        input_variables=["query"],
        template=template,
    )
