import gradio as gr
from pictionary_app.main import model


model.load("model_object.pt")


gr.Interface(
    fn=lambda img: img if img is None else model.predict(img),
    inputs="sketchpad",
    outputs="label",
    live=True,
    allow_flagging="never",
).launch()
