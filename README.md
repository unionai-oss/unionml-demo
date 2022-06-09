# UnionML demo

## Environment Setup

```
python -m venv ~/venvs/unionml-demo
source ~/venvs/unionml-demo/bin/activate
pip install -r requirements.txt -r requirements-demo.txt
```

## Deployment

To deploy the `unionml` app:

```
unionml deploy pictionary_app.main:model
```

This will upload an image to the [unionml repo](https://github.com/unionai-oss/unionml/pkgs/container/unionml).
For the purposes of this demo, we'll use the same image for the flytekit demo, since it packages all the
source code in this directory.

To deploy the flytekit demo:

```
pyflyte --pkgs flytekit_demo package --image "ghcr.io/unionai-oss/unionml:<tag>"
flytectl register files -c ./config/config-remote.yaml --project unionml --domain development --archive flyte-package.tgz --version <version>
```

## Start the Demo

Start a jupyter lab server:

```
jupyter lab
```

Go to jupyter lab on your browser and open `demo.ipynb` and run the code!
