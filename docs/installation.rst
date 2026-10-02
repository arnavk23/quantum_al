Installation
============

``quantum_al`` requires Python 3.10 or newer. The core package depends only
on NumPy, SciPy and scikit-learn.

.. code-block:: bash

   git clone https://github.com/arnavk23/quantum_al.git
   cd quantum_al
   python -m venv .venv
   source .venv/bin/activate          # .venv\Scripts\activate on Windows
   pip install -e .

Optional extras:

=============  ===============================================================
Extra          Needed for
=============  ===============================================================
``data``       downloading Materials Project data (``pymatgen``)
``circuit``    the Qiskit circuit realization (``qiskit``, ``qiskit-aer``)
``plot``       regenerating the paper figures (``matplotlib``)
``test``       running the test suite (``pytest``, ``pytest-cov``)
``docs``       building this documentation (``sphinx``)
``all``        everything above
=============  ===============================================================

.. code-block:: bash

   pip install -e ".[all]"
   pytest                         # 100+ tests, about 15 s

Materials Project data
----------------------

The real-data benchmarks use DFT-computed properties from the Materials
Project. The data are not redistributed in the repository; fetch them once
with a free API key from https://next-gen.materialsproject.org/api:

.. code-block:: bash

   export MP_API_KEY=your_key
   python -m quantum_al.fetch_data            # writes data/<task>.json

Files are written to (and read from) ``data/`` in a source checkout, or to
the directory named by the ``QUANTUM_AL_DATA_DIR`` environment variable.
Everything except the real-data benchmarks and
``examples/05_materials_project.py`` works without the data.
