

## Typing annotations

### Two ways declaring the type

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

```python
```

```python
```