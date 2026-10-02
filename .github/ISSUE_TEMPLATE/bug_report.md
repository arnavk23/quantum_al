---
name: Bug report
about: Something does not work, or a result does not reproduce
labels: bug
---

**What happened**
A clear description of the problem. For a result that does not reproduce,
name the script, the `results/` file and the number that differs.

**How to reproduce**
```bash
# exact commands
```

**Expected behavior**

**Environment**
- `quantum_al` version (`python -c "import quantum_al; print(quantum_al.__version__)"`):
- Python version and OS:
- Output of `pip freeze | grep -iE "numpy|scipy|scikit-learn|qiskit|pymatgen"`:

**Does `pytest` pass on your machine?**
