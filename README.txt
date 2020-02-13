  # The Welfare State Analytics Repository


  ### Prerequisites

  ### Installation

  ```bash
 % pip install pipenv
 % git clone https://github.com/humlab/welfare_state_analytics.git
 % cd welfare_state_analytics/
 % xcode-select --install
 % pip install cytoolz
 % pipenv install
 % jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyterlab_bokeh
```

### Note

Unit test debugging with `pytest` only works when code-coverage is disabled:

```
{
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "--no-cov"
    ]
}

{
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "--cov-config=.coveragerc",
        "--cov=./westac",
    ]
}

```

See [issue](https://github.com/microsoft/vscode-python/issues/693) and [issue](https://github.com/kondratyev-nv/vscode-python-test-adapter/issues/123).

