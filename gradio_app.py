from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import json
import gradio as gr
from concurrent.futures import ThreadPoolExecutor
import re
import pandas as pd
from prompter import scoring_prompt, format_prompt

def llm(messages, model="gpt-4-turbo-preview", json_mode=False):
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    system = [{"role": "system", "content": "You are a helpful assistant designed to output JSON."}] if json_mode else []
    response = OpenAI().chat.completions.create(
        model=model,
        # model="gpt-3.5-turbo",
        messages=system+messages,
        max_tokens=4096,
        temperature=0,
        **kwargs
    )
    response = response.choices[0].message.content.replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
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
        try:
            formatted_json = json.dumps(json.loads(json_string), indent=4)
            return s.replace(json_string, formatted_json)
        except json.decoder.JSONDecodeError as e:
            print(f"Error decoding JSON: {e}")
            print(s)
            return s
    return s

def fn_batch(prompt, df, file=None):
    vars = re.findall(r'\{\{([^{}]*?)\}\}', prompt)
    data = df.to_dict('records')
    for v in vars:
        if not all(v in d for d in data):
            raise gr.Error("Prompt variables must match column headers in data.")
    responses = []
    with ThreadPoolExecutor(max_workers=40) as executor:
        for d in data:
            new_prompt = prompt
            for k, v in d.items():
                new_prompt = new_prompt.replace("{{"+k+"}}", str(v))
            responses.append(executor.submit(llm, [{"role": "user", "content": new_prompt}]))
    responses = [r.result() for r in responses]
    df['response'] = [prettify_json_string(r) for r in responses]
    if file:
        df.to_csv(file, index=False)
    return gr.DataFrame(df, interactive=True, datatype=["str"] * (len(df.columns) - 1) + ["markdown"])

def fn_upload(file):
    if file is None:
        return [gr.DataFrame(visible=False), gr.DownloadButton(visible=False)]
    return [gr.DataFrame(pd.read_csv(file), visible=True), gr.DownloadButton(visible=True, value=file)]

def fn_auto(df, prompt):
    system = """Design a new prompt for GPT-4 that responds to the following inputs with the corresponding outputs exactly.
    
A good prompt generally has 3 elements:
1. A clear instruction for the model to follow.
2. Good examples of inputs and outputs.
3. The output format.

Only respond with the final prompt."""
    if not prompt:
        for i, row in df.iterrows():
            system += f"\n\nExample {i+1}:\nInput:\n{row['input']}\nOutput:\n{row['output']}"
    else:
        system += """\n\nWith reference to the current prompt and how the generated responses are different from the corresponding outputs, design a new prompt that better guides the model to generate the exact outputs.

It may be useful to specify what the model should do or not do.

Only respond with the new prompt."""
        system += f"\n\nPrevious Prompt:\n{prompt}"
        for i, row in df.iterrows():
            system += f"\n\nExample {i+1}:\nInput:\n{row['input']}\nOutput:\n{row['output']}\nResponse:\n{row['response']}"
    prompt = llm([{"role": "user", "content": system}])
    data = df.to_dict('records')
    responses = []
    with ThreadPoolExecutor(max_workers=40) as executor:
        for d in data:
            responses.append(executor.submit(llm, [{"role": "user", "content": prompt + "\n\nInput:\n{input}\n\nOutput:\n".format(**d)}]))
    responses = [r.result() for r in responses]
    response = [prettify_json_string(r) for r in responses]
    df['response'] = response
    return [gr.Textbox(prompt, visible=True), gr.DataFrame(df)]

def fn_prompter(initial_prompt):
    return format_prompt(json.loads(initial_prompt))

with gr.Blocks() as demo:
    gr.Markdown("# 🕶 EasyLLM")

    with gr.Tab(label="PromptFormatter"):
        initial_prompt = gr.Textbox(label="Prompt", value=json.dumps(scoring_prompt, indent=4))
        run = gr.Button("🚀 Run")
        formatted_prompt = gr.Markdown()
        run.click(fn=fn_prompter, inputs=initial_prompt, outputs=formatted_prompt)

    with gr.Tab(label="BatchPrompter"):
        prompt = gr.Textbox(label="Prompt", value="You are a friendly AI.\n\nReply to {{name}}'s message:\n\n{{message}}")
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

    with gr.Tab(label="AutoPrompter"):
        df = gr.DataFrame(pd.DataFrame({
            "input": ["The quick brown fox jumped over the lazy dog", "That was a hell of an amazing movie, man.", "What the hell is this? This movie sucks!"],
            "output": ["NEUTRAL", "POSITIVE", "NEGATIVE"]
            }),
            interactive=True,
            datatype="markdown"
        )
        prompt = gr.Textbox(label="Prompt", value="")
        run = gr.Button("🚀 Run")
        run.click(fn=fn_auto, inputs=[df, prompt], outputs=[prompt, df])

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=40)
    demo.launch()