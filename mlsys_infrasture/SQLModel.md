# SQLModel
SQLModel is an ORM for Python that aims to be developer-friendly and simple to use. It is built on top of SQLAlchemy, and provides a declarative syntax for defining database models.

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
# Insert 
# Select 
# Filter
# Update 
# Delete
```

## Installation

```bash
pip install sqlmodel # assume you have installed python>=3.9
sudo apt install sqlite3 # install sqlite3
sudo apt-get install sqlitebrowser # install db browser
```

## Use Cases

```python
# Create a table
# Insert 
# Select 
# Filter
# Update 
# Delete
```

## Examples

```bash
# File Structure
.
├── models.py
├── schema.py
├── main.py
└── tests
```

```python
# models.py
```