How to build the documentation
------------------------------

To build the documentation, we require [sphinx](https://www.sphinx-doc.org/) with the following plugins:

- [nbsphinx](https://nbsphinx.readthedocs.io/) for rendering of jupyter notebooks
- [sphinxcontrib-bibtex](https://sphinxcontrib-bibtex.readthedocs.io/) for reference management
- [matplotlib](https://matplotlib.org/) for plotting
- [sphinxcontrib-katex](https://sphinxcontrib-katex.readthedocs.io/) for equation rendering:
    - we do server-side rendering of equations, meaning that `KaTeX` has to be installed as an executable via 
      ``yarn global add katex`` or ``npm install -g katex``.
    - to disable server-side rendering, change
      ```python 
      katex_prerender = True
      ```
      to
      ```python
      katex_prerender = False
      ```
      in `source/conf.py`
- [sphinx-gallery](https://sphinx-gallery.github.io/) for example galleries

Further requirements are a working installation of [jupyter notebook](https://jupyter.org/) and deeptime.

Once all requirements are satisfied, a call to
```shell script
make html
```
from the `docs` directory builds the documentation, the output can be found under `docs/builds/html`.

Working with references
-----------------------
If you want to document something in the library and work with references, there is a global 
`docs/source/references.bib` bibTeX file. Every page with documentation needs its own bibTeX label prefix to not
confuse Sphinx' bookkeeping. For example, in the case of k-means, you can find
```
References
----------
.. bibliography:: /references.bib
    :style: unsrt
    :filter: docname in docnames
    :keyprefix: kmeans-
```
in the docstring, meaning that all bibtex keys are prefixed with `kmeans-`. To actually cite something in ReST,
one can use the `:cite:` directive:
```
[...] For details, see :cite:`kmeans-arthur2006k`.
```
This pulls the `arthur2006k` reference from the global `references.bib` file.

Building the documentation with notebooks
-----------------------------------------

Building the documentation can take a while, especially when KaTeX prerendering and jupyter notebook to html
conversion is enabled. For this reason, the default behavior is **no** prerendering and also **no** notebooks.
To enable both of them, please invoke

```shell script
make html SPHINXOPTS="-t notebooks"
``` 

If you want to execute or clear all notebooks then you can use the respective `run.sh` and `clear.sh` bash scripts. 
