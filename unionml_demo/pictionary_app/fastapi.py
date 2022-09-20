from fastapi import FastAPI
from pictionary_app.main import model


app = FastAPI()
model.serve(app, remote=True, model_version="f43354ef15d174fd7b85")
