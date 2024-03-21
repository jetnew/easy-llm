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
        model=model,
        # model="gpt-3.5-turbo",
        messages=system+messages,
        max_tokens=4096,
        **kwargs
    )
    response = response.choices[0].message.content
    if json_mode:
        response = json.loads(response)
    return response

def extract_json_string(s):
    open_braces, json_start_index = 0, s.find('{')
    if json_start_index == -1: return None
    for i in range(json_start_index, len(s)):
        open_braces += (s[i] == '{') - (s[i] == '}')
        if open_braces == 0: return s[json_start_index:i+1]
    return None

def prettify_json_string(s):
    json_string = extract_json_string(s)
    if json_string:
        return s.replace(json_string, json.dumps(json.loads(json_string), indent=4))
    return s

def fn_batch(prompt, df, file=None):
    vars = re.findall(r'\{([^{}]*?)\}', prompt.replace('{{', '!@#').replace('}}', '#@!'))
    data = df.to_dict('records')
    for v in vars:
        if not all(v in d for d in data):
            raise gr.Error("Prompt variables must match column headers in data.")
    with ThreadPoolExecutor(max_workers=30) as executor:
        responses = [executor.submit(llm, [{"role": "user", "content": prompt.format(**d)}]).result() for d in data]
    df['response'] = [prettify_json_string(r) for r in responses]
    if file:
        df.to_csv(file, index=False)
    return gr.DataFrame(df, interactive=True, datatype=["str"] * (len(df.columns) - 1) + ["markdown"])

def fn_upload(file):
    if file is None:
        return [gr.DataFrame(visible=False), gr.DownloadButton(visible=False)]
    return [gr.DataFrame(pd.read_csv(file), visible=True), gr.DownloadButton(visible=True, value=file)]

def fn_auto(df, prompt):
    system = """Design a new prompt for GPT-4 that responds to the following inputs with the corresponding outputs.
    
A good prompt generally has 3 elements:
1. A clear instruction for the model to follow.
2. Good examples of inputs and outputs.
3. The output format.

Only respond with the final prompt."""
    if not prompt:
        for i, row in df.iterrows():
            system += f"\n\nExample {i+1}:\nInput:\n{row['input']}\nOutput:\n{row['output']}"
    else:
        system += """\n\nWith reference to the previous prompt and how the generated responses differ from the outputs, design a new prompt that better guides the model to generate the correct outputs.

It may be useful to specify what the model should do or not do.

Only respond with the new prompt."""
        system += f"\n\nPrevious Prompt:\n{prompt}"
        for i, row in df.iterrows():
            system += f"\n\nExample {i+1}:\nInput:\n{row['input']}\nOutput:\n{row['output']}\nResponse:\n{row['response']}"
    print(system)
    prompt = llm([{"role": "user", "content": system}])
    data = df.to_dict('records')
    with ThreadPoolExecutor(max_workers=30) as executor:
        responses = [executor.submit(llm, [{"role": "user", "content": (prompt + "\n\nInput:\n{input}\n\nOutput:\n").format(**d)}]).result() for d in data]
    response = [prettify_json_string(r) for r in responses]
    df['response'] = response
    return [gr.Textbox(prompt, visible=True), gr.DataFrame(df)]

with gr.Blocks() as demo:
    gr.Markdown("# 🕶 EasyLLM")

    with gr.Tab(label="AutoPrompter"):
        df = gr.DataFrame(pd.DataFrame({
            "input": ["John, Data Scientist, Google", "Jane, Software Engineer, Facebook"],
            "output": [
                "Score: 10/10\nExplanation: As a data scientist at Google, John is highly qualified for the job opening of data scientist at Microsoft.",
                "Score: 5/10\nExplanation: Although Facebook is a prestigious tech company, Jane's experience as a software engineer is not directly relevant to the data scientist role at Microsoft."]
            }),
            interactive=True,
            datatype="markdown"
        )
        prompt = gr.Textbox(label="Prompt", value="")
        run = gr.Button("🚀 Run")
        run.click(fn=fn_auto, inputs=[df, prompt], outputs=[prompt, df])

    with gr.Tab(label="BatchPrompter"):
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
            run.click(fn=fn_batch, inputs=[prompt, df, file], outputs=df)
            download = gr.DownloadButton(label="⬇️ Download", visible=False)
        file.change(fn=fn_upload, inputs=file, outputs=[df, download])

if __name__ == "__main__":
    demo.launch()