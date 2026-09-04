Installation notes

If `pip install -r requirements.txt` fails while downloading large packages (e.g., torch), you have several options.

1) Install only the lightweight/core requirements (recommended for running tests and the simulation without heavy models):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) If you need the full ML stack (PyTorch + torchvision), install the ML extras separately. Options:

- Use the bundled `requirements_ml.txt`:

```bash
pip install -r requirements_ml.txt
```

- Or follow PyTorch's official install page to pick the correct wheel for your CUDA/Python setup: https://pytorch.org/get-started/locally/

3) If downloads time out, try one of:

- Increase pip's timeout temporarily:

```bash
pip --default-timeout=1000 install -r requirements_ml.txt
```

- Use a faster connection or download a wheel manually and install with `pip install /path/to/torch.whl`.

- Use conda (recommended for many systems):

```bash
conda create -n swarm python=3.10
conda activate swarm
conda install pytorch torchvision -c pytorch
pip install -r requirements.txt
```

4) For CI or reproducible runs, pin exact wheel URLs or use manylinux wheels saved locally.

If you'd like, I can:
- Attempt to install only the lightweight requirements here and run the tests.
- Or try `pip install --default-timeout=1000 -r requirements_ml.txt` for you.

Which should I try next?