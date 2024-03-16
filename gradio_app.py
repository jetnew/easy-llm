from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import json
import gradio as gr
from concurrent.futures import ThreadPoolExecutor
import re

def llm(messages, model="gpt-4-turbo-preview", json_mode=False):
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    system = [{"role": "system", "content": "You are a helpful assistant designed to output JSON."}] if json_mode else []
    response = OpenAI().chat.completions.create(
        model=model,
        messages=system+messages,
        max_tokens=4096,
        **kwargs
    )
    response = response.choices[0].message.content
    if json_mode:
        response = json.loads(response)
    return response

def fn(prompt, data):
    vars = re.findall(r'\{([^{}]*?)\}', prompt)
    prompt = prompt.replace('{{', '{').replace('}}', '}')
    data = data.to_dict('records')
    for v in vars:
        if not all(v in d for d in data):
            raise gr.Error("Prompt variables must match column headers in data.")
    futures = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        for d in data:
            futures.append(executor.submit(llm, [{"role": "user", "content": prompt.format(**d)}]))
    for d, f in zip(data, futures):
        d["response"] = f.result()
    return data

with gr.Blocks() as demo:
    prompt = gr.Textbox(label="Prompt", value="You are a friendly AI.\n\nReply to {name}'s message:\n\n{message}")
    with gr.Tab("Data"):
        data = gr.Dataframe(
            value=[
                ["John", "Hey there!"],
                ["Jane", "What's the weather today?"]
            ],
            headers=["name", "message"],
            interactive=True,
        )
    with gr.Tab("Upload"):
        gr.File(label="Data")
    btn = gr.Button("🚀 Run")
    response = gr.JSON(label="Response")
    btn.click(fn=fn, inputs=[prompt, data], outputs=response)


if __name__ == "__main__":
    demo.launch()