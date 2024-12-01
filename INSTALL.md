# Installation

You can install Haverscript directly from the GitHub repository using `pip`.

Here's how to set up Haverscript:

1. First, create and activate a Python virtual environment if you haven’t already:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install Haverscript directly from the GitHub repository:

```bash
pip install git+https://github.com/andygill/haverscript.git@v0.1.0
```

In the future, if there’s enough interest, I plan to push Haverscript to PyPI
for easier installation.

# Download and install

If you are a HaverScript user, then the commands above should work for you. 

However,
if you want to make changes to HaverScript, you will need to download the
repo, and build by hand. I use the following.

```bash
git clone git@github.com:andygill/haverscript.git
cd haverscript
python3 -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install -e .
```

Now any local example can be run directly.

```shell
python examples/first_example/main.py
```

```markdown
> In one sentence, why is the sky blue?

The sky appears blue due to scattering of sunlight by molecules and particles in the Earth's atmosphere.

...
```

# Testing

The unit tests are code fragments to test specific features
The e2e tests execute a sub-process, testing the examples.

```
pytest tests      # run all tests
pytest tests/unit # run unit tests
pytest tests/e2e  # run e2e tests
```

The e2e tests uses `pytest-regressions`. The golden output is version controlled. 

We use the following models:
* ollama mistral:v0.3 (f974a74358d6)
* together.ai meta-llama/Meta-Llama-3-8B-Instruct-Lite
* ollama llava:v1.6 (8dd30f6b0cb1)

The tests need a valid TOGETHER_API_KEY in the environment when run.

If you need to regenerate test output, use 

```
pytest tests/e2e --force-regen
```
