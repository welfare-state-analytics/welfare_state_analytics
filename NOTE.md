

# Typing annotations

## The two ways of declaring the type

```python
x : int = 1
x = 1        # type: int
```

| Type          | Syntax                                       | Example
| -----         | -----                                        | -------
| Simple types  | x : `int`                                    | x: int = 1
|               | x : `float`                                  | x: float = 1.0
|               | x : `bool`                                   | x: bool = True
|               | x : `str`                                    | x: str = "test"
|               | x : `bytes`                                  | x: bytes = b"test"
| Collections   | x : `List[`_type_`]`                         | x: List[int] = [1]
|               | x : `Set[`_type_`]`                          | x: Set[int] = {6, 7}
| Dict          | x : `Dict[`_type_, _type_`]`                 | x: Dict[str, int] = { "a": 6 }
| Tuple         | x : `Tuple[`_type_, ...`]`                   | x: Tuple[int,floar] = {6, 7.0}
| Optional      | x : `Optional[`_type_`]`                     | x: Optional[int] = None
| Callable      | x : `Callable[`_from-types_`,` _to-types_`]` | x: Callable[[int, float], float] = fx
| Iterator      | x : `Iterator[`_type_`]`                     | x: Iterator[int] = range(1, 100)

### Type alias

```python
from typing import List, Tuple

Point = Tuple[int, int]

points: List[Point] = [ (1,2) ]
```

```python
from typing import Tuple

def get_api_response() -> Tuple[int, int]:
    successes, errors = ... # Some API call
    return successes, errors
```

```python
from typing import Union

def print_grade(grade: Union[int, str]):
    if isinstance(grade, str):
        print(grade + ' percent')
    else:
        print(str(grade) + '%')
```

```python
# merge to dicts
x = { 'a': 1, 'b': 2 }
y = { 'b': 4, 'c': 5 }

z = {**x, **y}

```

#### Install Git-LFS on Debian / Ubuntu

```bash
% sudo apt-get update
% sudo apt-get upgrade
% curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
% git lfs install
```

#### Pre-commit Hook that clears Notebook Outputs

Source: https://zhauniarovich.com/post/2020/2020-06-clearing-jupyter-output/

Install `pre-commit`:

```bash
% pipenv install pre-commit --dev
```

Create a new file names `.pre-commit-config.yml` and with the following text:

```yaml
repos:
  - repo: local
    hooks:
      - id: jupyter-nb-clear-output
        name: jupyter-nb-clear-output
        files: \.ipynb$
        stages: [commit]
        language: system
        entry: jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace
```

Install the hook:

```bash
% pipenv shell
% pre-commit install
```

## Note

Unit test debugging with `pytest` only works when code-coverage is disabled:

```json
{
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [ "--no-cov" ]
}
{
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [ "--cov-config=.coveragerc", "--cov=./notebooks", ]
}
```

See [issue](https://github.com/microsoft/vscode-python/issues/693) and [issue](https://github.com/kondratyev-nv/vscode-python-test-adapter/issues/123).
