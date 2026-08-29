from langchain.tools import tool

@tool(description="用以查询某个地方，某个日期的天气的一个工具")
def check_weather(weather_date:str, check_city:str):

    return f"{check_city}的天气在{weather_date}是晴朗的"


from langchain_openai import ChatOpenAI
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s',level=logging.DEBUG)
llm = ChatOpenAI(
    base_url="https://api.deepseek.com",
    api_key="sk-c560a3bdaef045b7a959c830f324819c",
    model="deepseek-v4-flash"
)

llm_with_tools = llm.bind_tools(tools=[check_weather])

result = llm_with_tools.invoke([{"role":"user","content":"帮我查一下深圳8月30号的天气是怎么样的？"}])

print(result)
