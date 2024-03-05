# Python Programming

- Table of contents
  - [Typing Hints](#typing-hints)
  - [Comments](#comments)
  - [Unit Tests](#unit-tests)


## Typing Hints
https://fastapi.tiangolo.com/python-types/
https://mypy.readthedocs.io/en/latest/cheat_sheet_py3.html

```python
from typing import Optional

def greeting(name: Optional[str] = None) -> None:
    if name is not None:
        print(f"Hello, {name}!")
    else:
        print("Hello, Stranger!")
```

## Comments
```python
from typing import Dict

def greeting(cutomers: Dict[str, str]) -> None:
    """ 
    print greeting message
    
    arguments:
    	customers: user data
    
    returns:
    	none
    
    usage:
    	users = dict({'jason': '123', 'marry': '222'})
    	greeting(users)
    """
    customer_names = customers.keys()
    for customer in customer_names:
        print(f"Hello {customer}, Your ID is {customers[customer]}.")
```

## Unit Test

```python
# test_parameterize.py
# pytest ./test_parameterize.py
@pytest.mark.parametrize('passwd', ['1234', 'jasonqww'])
def test_passwd_length(passwd):
    assert len(passwd) >= 8
```