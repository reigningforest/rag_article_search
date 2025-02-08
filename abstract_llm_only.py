import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI


def get_env():
    '''Load environment variables'''
    current_file_dir = os.path.abspath(os.path.dirname(__file__))
    env_path = os.path.join(current_file_dir, '.env')
    load_dotenv(dotenv_path=env_path)


def loading():
    # Load the environment variables
    get_env()

    # Create LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    print('LLM loaded')
    return llm


def query_llm(llm, query_str):
    '''Query the LLM'''
    response = llm.invoke(query_str)
    return response.content


def main():
    # Load the environment variables
    get_env()

    # Create LLM
    llm = loading()

    # Query the LLM
    print("Enter 'exit' to quit the program.")
    while True:
        # Allow for multiple question cycles
        query_str = input("Enter a question: ")
        if query_str == "exit":
            break
        answer = query_llm(llm, query_str)
        print(answer)


if __name__ == "__main__":
    main()