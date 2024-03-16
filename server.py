import gradio as gr
import uvicorn
from fastapi import FastAPI
from gradio_app import gr_app

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Hello World"}

app = gr.mount_gradio_app(app, gr_app, path="/")


if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)