# Python Programming

## Typing Hints
https://fastapi.tiangolo.com/python-types/
https://mypy.readthedocs.io/en/latest/cheat_sheet_py3.html

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