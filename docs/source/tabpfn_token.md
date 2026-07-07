# TabPFN Token Setup

TabPFN requires a one-time Prior Labs license acceptance before it can download
model weights for local inference. STREAMLINE can import and configure TabPFN
without this token, but Phase 6 will skip requested TabPFN models until
`TABPFN_TOKEN` is available in the environment.

```{warning}
If `TABPFN_TOKEN` is not set, Phase 6 warns and skips requested TabPFN models
instead of failing the full run. Other requested models, including HEROS, still
run normally. A passing test suite with skipped TabPFN tests means TabPFN was
not fully run.
```

## Get A Token

1. Open `https://ux.priorlabs.ai` in a browser.
2. Log in or create an account.
3. Accept the TabPFN license on the Licenses tab.
4. Copy the API key from `https://ux.priorlabs.ai/account`.

Do not commit this token to git, notebooks, config files, shell history shared
with others, or issue trackers.

## Set The Token For One Terminal Session

Activate your STREAMLINE environment and export the token before running
TabPFN models or TabPFN fit tests:

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

## Run The Tests

The default pytest suite runs the main binary, multiclass, and regression
end-to-end tests:

```bash
pytest
```

Phase-level subtests are kept for targeted maintainer debugging and are skipped
by default collection. With `TABPFN_TOKEN` set, maintainers can run the
token-gated TabPFN subtest by overriding the default pytest `addopts`:

```bash
pytest -o addopts='' streamline/tests/subtests/test_p6_heros_tabpfn.py -q
```

Without `TABPFN_TOKEN`, TabPFN fit checks are intentionally skipped and Phase 6
warns when TabPFN is requested. Other requested models continue to run.

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
