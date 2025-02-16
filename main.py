from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI

load_dotenv()

llm = OpenAI()
response = llm.invoke("Hello how are you?")
print(response)

prompt = PromptTemplate.from_template("How to say {input} in {output_language}:\n")

chain = prompt | llm
response = chain.invoke(
    {
        "output_language": "German",
        "input": "I love programming.",
    }
)

print(response)
