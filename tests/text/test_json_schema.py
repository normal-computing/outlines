from enum import Enum

from pydantic import BaseModel

import outlines.text.json_schema as json_schema


def test_pydantic_base():
    class User(BaseModel):
        user_id: int
        name: str

    schema = User.schema_json()
    schedule = json_schema.build_schedule_from_schema(schema)
    assert schedule == ['{\n  "user_id": ', 1, ',\n  "name": ', None, ",\n}"]


def test_pydantic_float():
    class User(BaseModel):
        user_id: int
        value: float

    schema = User.schema_json()
    schedule = json_schema.build_schedule_from_schema(schema)
    assert schedule == ['{\n  "user_id": ', 1, ',\n  "value": ', 1.3, ",\n}"]


def test_pydantic_enum():
    class Name(str, Enum):
        john = "John"
        marc = "Marc"
        michel = "Michel"

    class User(BaseModel):
        user_id: int
        name: Name

    schema = User.schema_json()
    schedule = json_schema.build_schedule_from_schema(schema)
    assert schedule == [
        '{\n  "user_id": ',
        1,
        ',\n  "name": ',
        ["John", "Marc", "Michel"],
        ",\n}",
    ]


def test_pydantic_nested():
    """Arbitrarily nested schema."""
    from typing import List

    class Fizz(BaseModel):
        buzz: str

    class Foo(BaseModel):
        count: int
        size: Fizz

    class Bar(BaseModel):
        apple: str
        banana: str

    class Spam(BaseModel):
        foo: Foo
        bars: List[Bar]

    # We need to a recursive function to parse nested schemas
    schema = Spam.schema_json()
    schedule = json_schema.build_schedule_from_schema(schema)
    assert schedule == [
        '{\n  "foo": {\n    "count": ',
        1,
        ',\n    "size": {\n      "buzz": ',
        None,
        ',\n    },\n  },\n  "bars": [{\n    "apple": ',
        None,
        ',\n    "banana": ',
        None,
        ",\n  }],\n}",
    ]


def test_pydantic_union():
    """Schemas with Union types."""
    from typing import Union

    class Spam(BaseModel):
        foo: int
        bar: Union[float, str]

    schema = Spam.schema_json()
    schedule = json_schema.build_schedule_from_schema(schema)
    assert schedule == [
        '{\n  "foo": ',
        1,
        ',\n  "bar": ',
        ["number", "string"],
        ",\n}",
    ]


def test_json_schema():
    schema = '{"title": "User", "type": "object", "properties": {"user_id": {"title": "User Id", "type": "integer"}, "name": {"title": "Name", "type": "string"}}, "required": ["user_id", "name"]}'
    schedule = json_schema.build_schedule_from_schema(schema)
    assert schedule == ['{\n  "user_id": ', 1, ',\n  "name": ', None, ",\n}"]
