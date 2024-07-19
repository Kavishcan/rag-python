from get_embedding_function import get_embedding_function
import os
import argparse
# from langchain.vectorstores.chroma import Chroma
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from openai import OpenAI

client = OpenAI(
    api_key="sk-proj-JuGeFyd7UOH9Ly37Qkm8T3BlbkFJjxQvOFOfj9aAOD5yEs0K")

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context and give the answer in rich editor : {question}
"""
# Read the input paragraph and genrate multiple choice question and answers in the paragraph based on the above context : {question}


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    print(f"Querying: {query_text}")
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH,
                embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    correctPrompt = prompt_template.format(
        context=context_text, question=query_text)

    prompt = {
        "role": "user",
        "content": correctPrompt,
    }

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[prompt],
        temperature=0.0,
    )

    response_text = response.choices[0].message.content

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text


if __name__ == "__main__":
    main()
