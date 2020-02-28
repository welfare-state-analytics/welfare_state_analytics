# The Welfare State Analytics Repository

### Prerequisites

### Installation

See [this page](https://github.com/humlab/welfare_state_analytics/wiki/How-to:-Install-notebooks-on-local-machine).

### Note

Unit test debugging with `pytest` only works when code-coverage is disabled:
```
{
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [ "--no-cov" ]
}
{
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [ "--cov-config=.coveragerc", "--cov=./westac", ]
}
```
See [issue](https://github.com/microsoft/vscode-python/issues/693) and [issue](https://github.com/kondratyev-nv/vscode-python-test-adapter/issues/123).