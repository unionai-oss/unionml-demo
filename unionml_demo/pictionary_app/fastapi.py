from fastapi import FastAPI
from pictionary_app.main import model


app = FastAPI()
model.serve(
    app,
    remote=True,
    app_version="2a1644a10611c3f279f41f4e040eff6d04d8cc29",
    model_version="f43354ef15d174fd7b85",
)
