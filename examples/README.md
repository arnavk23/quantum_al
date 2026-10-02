# Examples

Runnable scripts, ordered from first contact to research use. Examples 01-04
need no data download or API key and each finishes in about a minute or less
on a laptop; they are run in CI on every push.

| Script | What it shows |
|---|---|
| `01_quickstart.py` | Joint vs. correlation-blind acquisition on a two-property problem, with paired statistics |
| `02_information_decomposition.py` | The exact identities behind the joint EIG: decomposition into marginal terms minus total correlation, the sandwich bound, unit invariance, and the second-order behavior of the correlation term |
| `03_batch_selection.py` | Why naive top-k batches waste labels, and how greedy batch EIG (with its (1 - 1/e) guarantee) avoids it |
| `04_custom_selector.py` | Plugging your own acquisition function into the harness and comparing it to the nine classical baselines with Holm-Bonferroni correction |
| `05_materials_project.py` | The same workflow on real Materials Project data (requires `python -m quantum_al.fetch_data` and an API key) |

Run from the repository root after `pip install -e .`:

```bash
python examples/01_quickstart.py
```
