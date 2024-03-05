# SQLModel
SQLModel is an ORM for Python that aims to be developer-friendly and simple to use. It is built on top of SQLAlchemy, and provides a declarative syntax for defining database models.

**Offciial Doc**: https://sqlmodel.tiangolo.com/

In this tutorial, we use SQLite to build user database. _SQLite is a simple database in a single file and used in many applications. It can be viewed easily by DB Browser for SQLite._

- Table of contents
  - [SQL Background](#basic-sql)
  - [Installation](#installation)
  - [Use Cases](#use-cases)
  - [Examples](#examples)

## SQL Background
```SQL
# Create a table
CREATE TABLE hero (
        id INTEGER NOT NULL, 
        name VARCHAR(20) NOT NULL, 
        secret_name VARCHAR(20) NOT NULL, 
        super_power VARCHAR(8), 
        age INTEGER, 
        PRIMARY KEY (id), 
        UNIQUE (name)
)
# Insert a row
INSERT INTO hero (name, secret_name, super_power, age) 
VALUES ('Deadpond', 'Dive Wilson', 'immortal', None)
# Select all rows
SELECT *
FROM hero
# Filter
SELECT *
FROM hero
WHERE age IS NULL
# Update 
UPDATE hero
SET age=16
WHERE name = "Spider-Boy"
# Delete
DELETE
FROM hero
WHERE name = "Spider-Youngster"
```

## Installation

```bash
pip install sqlmodel # assume you have installed python>=3.9
sudo apt install sqlite3 # install sqlite3
sudo apt-get install sqlitebrowser # install db browser
```

## Use Cases

```python
# Create a table named hero
import enum
from typing import Optional
from sqlmodel import Field, Enum, Column, String, SQLModel, create_engine, Session, select


class Superpower(str, enum.Enum):
    money = "rich"
    fly = "flying"
    immortal = "immortal"


class Hero(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str = Field(sa_column=Column(String(20), nullable=False, unique=True)) # unique
    secret_name: str = Field(sa_column=Column(String(20), nullable=False))
    super_power: Superpower = Field(sa_column=Column(Enum(Superpower)), default=Superpower.money)
    age: Optional[int] = None


sqlite_file_name = "database.db"

sqlite_url = f"sqlite:///{sqlite_file_name}"

engine = create_engine(sqlite_url, echo=True)

SQLModel.metadata.create_all(engine)

# Insert new heros
def create_heros(): 
    hero_1 = Hero(name="Deadpond", secret_name="Dive Wilson", super_power=Superpower.immortal)
    hero_2 = Hero(name="Rubber-Man", secret_name="Henry Philip")
    hero_list = [hero_1, hero_2]

    with Session(engine) as session:
        for hero in hero_list:
            try:
                session.add(hero)
                session.commit()
            except:
                print(f"****{hero.name} already exists****")
# Select all heros
def select_heroes():
    with Session(engine) as session:
        statement = select(Hero)
        results = session.exec(statement)
        for hero in results:
            print(hero)
# Find all heros (condition)
def select_hero_condition():
    with Session(engine) as session:
        statement = select(Hero).where(Hero.name == "Deadpond")
        results = session.exec(statement)
        for hero in results:
            print(hero)
# Update 
def update_heroes():
    with Session(engine) as session:
        statement = select(Hero).where(Hero.name == "Spider-Boy")
        results = session.exec(statement)
        hero = results.one()
        print("Hero:", hero)

        hero.age = 16
        session.add(hero)
        session.commit()
        session.refresh(hero)
        print("Updated hero:", hero)
# Delete
def delete_heroes():
    with Session(engine) as session:
        statement = select(Hero).where(Hero.name == "Spider-Youngster")
        results = session.exec(statement)
        hero = results.one()
        print("Hero: ", hero)

        session.delete(hero)
        session.commit()

        print("Deleted hero:", hero)

        statement = select(Hero).where(Hero.name == "Spider-Youngster")
        results = session.exec(statement)
        hero = results.first()

        if hero is None:
            print("There's no hero named Spider-Youngster")
# Optimize (index, rerank) -> TBD
```

## Examples

```bash
# File Structure
.
├── project
    ├── __init__.py
    ├── app.py
    ├── database.py
    └── models.py
```

```python
# project/models.py
import enum
from typing import Optional
from sqlmodel import Field, Enum, Column, String, SQLModel

class UserRole(str, enum.Enum):
    user = "user"
    vip = "vip"
    admin = "admin"

    
class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(sa_column=Column(String(20), nullable=False, unique=True)) # unique
    password: str = Field(sa_column=Column(String(20), nullable=False))
    user_role: UserRole = Field(sa_column=Column(Enum(UserRole)), default=UserRole.user)


def main():
    pass


if __name__ == '__main__':
    main()
```

```python
# project/database.py
import logging
from typing import Optional
from sqlmodel import Field, Enum, Column, String, SQLModel, create_engine, Session, select

from models import User, UserRole

sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"
engine = create_engine(sqlite_url, echo=True)

logger = logging.getLogger()
logger.setLevel('DEBUG')

def create_table():
    SQLModel.metadata.create_all(engine)


def insert_user(username: str, password: str, user_role: Optional[UserRole] = None):
    if user_role is None:
        new_user = User(username=username, password=password)
    else:
        new_user = User(username=username, password=password, user_role=user_role)
    with Session(engine) as session:
        try:
            session.add(new_user)
            session.commit()
        except:
            print("username is existing")


def get_user(username: str):
    with Session(engine) as session:
        try:
            statement = select(User).where(User.username == username)
            user = session.exec(statement).one()
            # print(f"user: {user}")
            return user
        except:
            print("Invalid username")


def update_user(username: str, password: str):
    with Session(engine) as session:
        try:
            statement = select(User).where(User.username == username)
            user = session.exec(statement).one()
            # print(f"Before: {user}")
            user.password = password
            session.add(user)
            session.commit()
            session.refresh(user)
            # print("Updated user:", user)
        except:
            print("Invalid username")


def main():
    pass


if __name__ == '__main__':
    main()


```

```python
# project/app.py
import database

def register(username: str, password: str):
    database.insert_user(username=username, password=password)


def get_user(username: str):
    return database.get_user(username=username)


def change_password(username: str, new_password: str):
    database.update_user(username=username, password=new_password)


def main():
    database.create_table()
    print("register a new user")
    register(username="jackson", password="443333")
    print("return the user info")
    print(f"user: {get_user(username='jackson')}")
    change_password(username="jackson", new_password="123456")
    print(f"user (after changed): {get_user(username='jackson')}")


if __name__ == "__main__":
    main()
```