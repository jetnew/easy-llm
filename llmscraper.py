import json
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

def llm(messages, model="gpt-4-turbo-preview", json_mode=False):
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    system = [{"role": "system", "content": "You are a helpful assistant designed to output JSON."}] if json_mode else []
    response = OpenAI().chat.completions.create(
        model=model,
        messages=system+messages,
        max_tokens=4096,
        temperature=0,
        **kwargs
    )
    response = response.choices[0].message.content.replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
    if json_mode:
        response = json.loads(response)
    return response

prompt = """You are a web scraping expert. You are tasked to write Python code with BeautifulSoup4 to extract the text content in the following format:

{{
    "latest_experience_title": <title>,
    "latest_experience_company": <company>,
    "latest_experience_duration": <duration>,
}}

HTML:
{html}

You are a web scraping expert. You are tasked to write Python code with BeautifulSoup4 to extract the text content in the following format:

{{
    "latest_experience_title": <title>,
    "latest_experience_company": <company>,
    "latest_experience_duration": <duration>,
}}
"""

with open("data/cleaned.html", "r") as f:
    html = f.read()
response = llm([{"role": "user", "content": prompt.format(html=html)}])
print(response)


