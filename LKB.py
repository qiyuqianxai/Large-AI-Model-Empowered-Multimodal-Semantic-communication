# 导入ChatOpenAI，这是LangChain对ChatGPT API访问的抽象
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import openai
openai_api_key= "your openai api-key"
openai.api_base="https://api.closeai-proxy.xyz/v1"
chat = ChatOpenAI(temperature=0.0,openai_api_key=openai_api_key)
template = """
## There is a dict '{UserInfo}' which records the information of the user.
## Improve the expression of this sentence '{InputText}' so that the user can understand it well.
"""

def personalized_semantics(userinfo,inputtext):
    prompt_template = ChatPromptTemplate.from_template(template)
    customer_messages = prompt_template.format_messages(
                        UserInfo=userinfo,
                        InputText=inputtext)
    customer_response = chat(customer_messages)
    personalized_semantic = customer_response.content
    print("personalized_semantic: ",personalized_semantic)
    return personalized_semantic

if __name__ == '__main__':
    userInfo = {"name":"Mike","interests":"running","language":"English","identify":"student","gender":"male"}
    input_text = "one woman wrote in the course of the conversation to her friend."
    personalized_semantic = personalized_semantics(userInfo,input_text)

    userInfo = {"name": "Jane", "interests": "shopping", "language": "English", "identify": "student","gender":"female"}
    personalized_semantic = personalized_semantics(userInfo, personalized_semantic)