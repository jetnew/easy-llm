from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import json
import gradio as gr
from concurrent.futures import ThreadPoolExecutor
import re
import pandas as pd

def llm(messages, model="gpt-4-turbo-preview", json_mode=False):
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    system = [{"role": "system", "content": "You are a helpful assistant designed to output JSON."}] if json_mode else []
    response = OpenAI().chat.completions.create(
        # model=model,
        model="gpt-3.5-turbo",
        messages=system+messages,
        max_tokens=4096,
        **kwargs
    )
    response = response.choices[0].message.content
    if json_mode:
        response = json.loads(response)
    return response

def prettify_json_string(s):
    match = re.search(r'\{.*?\}', s, re.DOTALL) 
    if match is not None:
        json_str = match.group(0)
        return s.replace(json_str, json.dumps(json.loads(json_str), indent=4))
    return s

def fn_run(prompt, df, file):
    vars = re.findall(r'\{([^{}]*?)\}', prompt.replace('{{', '!@#').replace('}}', '#@!'))
    data = df.to_dict('records')
    for v in vars:
        if not all(v in d for d in data):
            raise gr.Error("Prompt variables must match column headers in data.")
    with ThreadPoolExecutor(max_workers=30) as executor:
        responses = [executor.submit(llm, [{"role": "user", "content": prompt.format(**d)}]).result() for d in data]
    df['response'] = [prettify_json_string(r) for r in responses]
    df.to_csv(file, index=False)
    return gr.DataFrame(df, interactive=True, datatype=["str"] * (len(df.columns) - 1) + ["markdown"])

def fn_upload(file):
    if file is None:
        return [gr.DataFrame(visible=False), gr.DownloadButton(visible=False)]
    return [gr.DataFrame(pd.read_csv(file), visible=True), gr.DownloadButton(visible=True, value=file)]

with gr.Blocks() as demo:
    prompt = gr.Textbox(label="Prompt", value="You are a friendly AI.\n\nReply to {name}'s message:\n\n{message}")
    file = gr.File(label="Data")
    df = gr.DataFrame(pd.DataFrame({
        "name": ["John", "Jane"],
        "message": ["Hey there!", "What's the weather today?"]
        }),
        interactive=True,
        visible=False
    )
    with gr.Row():
        run = gr.Button("🚀 Run")
        run.click(fn=fn_run, inputs=[prompt, df, file], outputs=df)
        download = gr.DownloadButton(label="⬇️ Download", visible=False)
    file.change(fn=fn_upload, inputs=file, outputs=[df, download])

if __name__ == "__main__":
    demo.launch()