# TabPFN Token Setup

TabPFN is optional in STREAMLINE. Install it separately when you want to run
TabPFN models:

```bash
pip install tabpfn
```

TabPFN requires a one-time Prior Labs license acceptance before it can download
model weights for local inference. STREAMLINE can import and configure TabPFN
without this token, but Phase 6 skips requested TabPFN models until
`TABPFN_TOKEN` is available in the environment.

```{warning}
If `TABPFN_TOKEN` is not set, Phase 6 warns and skips requested TabPFN models
instead of failing the full run. Other requested models, including HEROS, still
run normally. A passing run with skipped TabPFN means TabPFN was not fit.
```

## Get A Token

1. Open `https://ux.priorlabs.ai` in a browser.
2. Log in or create an account.
3. Accept the TabPFN license on the Licenses tab.
4. Copy the API key from `https://ux.priorlabs.ai/account`.

Do not commit this token to git, notebooks, config files, shared shell history,
or issue trackers.

## Set The Token For One Terminal Session

Activate your STREAMLINE environment and export the token before running
TabPFN models:

```bash
conda activate streamline
export TABPFN_TOKEN="paste-your-api-key-here"
```

Confirm Python can see it:

```bash
python -c "import os; print('TABPFN_TOKEN set:', bool(os.environ.get('TABPFN_TOKEN')))"
```

## Make The Token Persistent For A Conda Environment

Create conda activation scripts so the token is set whenever the environment is
activated and removed when it is deactivated:

```bash
conda activate streamline
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"
printf 'export TABPFN_TOKEN="paste-your-api-key-here"\n' > "$CONDA_PREFIX/etc/conda/activate.d/tabpfn_token.sh"
printf 'unset TABPFN_TOKEN\n' > "$CONDA_PREFIX/etc/conda/deactivate.d/tabpfn_token.sh"
chmod 600 "$CONDA_PREFIX/etc/conda/activate.d/tabpfn_token.sh"
```

Restart the terminal or run:

```bash
conda deactivate
conda activate streamline
```

## Run The TabPFN-Specific Pytest

The TabPFN pytest is intentionally explicit-only so the main test suite does
not run a heavyweight model download or fit by accident. Run it directly:

```bash
pytest streamline/tests/subtests/tabpfn_smoke.py -q -rs
```

Without `TABPFN_TOKEN`, the no-token skip behavior is tested and the actual
TabPFN fit test is skipped with a visible reason. With `TABPFN_TOKEN` set, the
same command also runs the TabPFN binary wrapper fit smoke test.

## Use TabPFN In Notebooks

For local notebooks, set the token before launching Jupyter from the same shell:

```bash
conda activate streamline
export TABPFN_TOKEN="paste-your-api-key-here"
jupyter notebook
```

For Colab, store the token in Colab secrets or set it in a private cell before
running TabPFN models:

```python
import os
os.environ["TABPFN_TOKEN"] = "paste-your-api-key-here"
```

Avoid saving notebooks with a real token embedded in a code cell.
