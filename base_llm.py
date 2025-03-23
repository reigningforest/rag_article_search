import yaml
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

def loading(config):
    # Create LLM
    llm = ChatOpenAI(model_name=config['llm_model_name'])
    print('LLM loaded')
    return llm


def query_llm(llm, query):
    '''Query the LLM'''
    response = llm.invoke(query)
    return response.content


def main():
    # Load the config file
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    # Load the environment variables
    load_dotenv(dotenv_path=config['env_file'])

    # Create LLM
    llm = loading(config)

    # Query the LLM
    print("Enter 'exit' to quit the program.")
    while True:
        # Allow for multiple query cycles
        query = input("Enter a query or type 'exit': ")
        if query == "exit":
            break
        answer = query_llm(llm, query)
        print(answer)


if __name__ == "__main__":
    main()