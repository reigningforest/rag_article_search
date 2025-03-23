import os
from dotenv import load_dotenv
import yaml

from langchain_openai import ChatOpenAI


def loading(llm_model_name):
    # Create LLM
    llm = ChatOpenAI(model_name=llm_model_name)
    print('LLM loaded')
    return llm


def query_llm(llm, query_str):
    '''Query the LLM'''
    response = llm.invoke(query_str)
    return response.content


def main():
    # Load the config variables
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Load the environment variables
    env_file_name = config['env_file']
    load_dotenv(dotenv_path=env_file_name)

    # Create LLM
    llm_model_name = config['llm_model_name']
    llm = loading(llm_model_name)

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