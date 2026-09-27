import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain.agents import create_agent

# Load environment variables from .env
load_dotenv()


def main():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("ERROR: GROQ_API_KEY is missing from your .env file!")
        return

    # Initialize Groq model
    model = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        groq_api_key=api_key
    )

    tools = []
    agent_executor = create_agent(model, tools)

    print("=== AI Assistant Online (Groq) ===")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        if not user_input:
            continue

        print("Assistant: ", end="", flush=True)

        try:
            response = agent_executor.invoke(
                {"messages": [HumanMessage(content=user_input)]}
            )

            messages = response.get("messages", [])
            if messages:
                print(messages[-1].content)
            print()

        except Exception as e:
            print(f"\nError from API: {e}\n")


if __name__ == "__main__":
    main()