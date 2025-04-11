def chatmodel_demo():
    from langchain.chat_models import init_chat_model
    from langchain_google_vertexai import ChatVertexAI

    model: ChatVertexAI = init_chat_model(
        model="gemini-2.0-flash-001", model_provider="google-vertexai"
    )

    from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

    messages = [
        SystemMessage("Translate the following from English to Chinese"),
        HumanMessage("Nice to meet you"),
    ]

    response_message: AIMessage = model.invoke(messages)

    print(response_message)


def prompt_template_demo():
    from langchain_core.prompts import ChatPromptTemplate

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "Translate the following from English to {language}"),
            ("user", "{text}"),
        ]
    )

    prompt = prompt_template.invoke(
        {
            "language": "Chinese",
            "text": "Nice to meet you",
        }
    )

    from langchain.chat_models import init_chat_model
    from langchain_google_vertexai import ChatVertexAI

    model: ChatVertexAI = init_chat_model(
        model="gemini-2.0-flash-001",
        model_provider="google-vertexai",
    )

    response_message = model.invoke(prompt)
    print(response_message)


if __name__ == "__main__":
    # chatmodel_demo()

    prompt_template_demo()
