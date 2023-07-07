import itertools
import json


def build_schedule_from_schema(schema: str):
    """Turn a JSON schema into a generation schedule.

    JSON Schema is a declarative language that allows to annotate JSON documents
    with types and descriptions. These schemas can be generated from any Python
    datastructure that has type annotation: namedtuples, dataclasses, Pydantic
    models. And by ensuring that the generation respects the schema we ensure
    that the output can be parsed into these objects.
    This function parses the provided schema and builds a generation schedule which
    mixes deterministic generation (fixed strings), and sampling with constraints.

    Parameters
    ----------
    schema
        A string that represents a JSON Schema.

    Returns
    -------
    A generation schedule, which is a list of sequence generation steps.

    References
    ----------
    .. [0] JSON Schema. https://json-schema.org/

    """
    schema = json.loads(schema)

    definitions = {}
    if "definitions" in schema:
        for definition, annotation in schema["definitions"].items():
            definitions[f"#/definitions/{definition}"] = annotation

    schema = expand_json_schema(schema, definitions)
    schedule = build_schedule_from_instance(schema)

    # Concatenate adjacent strings
    reduced_schedule = [
        x
        for cls, grp in itertools.groupby(schedule, type)
        for x in (("".join(grp),) if cls is str else grp)
    ]

    return reduced_schedule


def expand_json_schema(raw_schema, definitions):
    """Replace references by their value in the JSON Schema.

    This recursively follows the references to other schemas in case
    of nested models. Other schemas are stored under the "definitions"
    key in the schema of the top-level model.

    """
    expanded_properties = {}

    if "properties" in raw_schema:
        for name, value in raw_schema["properties"].items():
            if "$ref" in value:  # if item is a single element
                expanded_properties[name] = expand_json_schema(
                    definitions[value["$ref"]], definitions
                )
            elif "type" in value and value["type"] == "array":  # if item is a list
                expanded_properties[name] = value
                expanded_properties[name]["items"] = expand_json_schema(
                    definitions[value["items"]["$ref"]], definitions
                )
            else:
                expanded_properties[name] = value

        return {
            "title": raw_schema["title"],
            "type": raw_schema["type"],
            "properties": expanded_properties,
        }

    else:
        return raw_schema


def build_schedule_from_instance(instance, indent=0):
    """Build a schedule from a instance.

    This recursively follows the references to other instances.

    Parameters
    ----------
    instance
        An instance, can be the JSON schema itself.
    indent
        The current indentation level

    Returns
    -------
    A generation schedule in the form of a list of `SequenceGenerator` instances.

    """
    schedule = []
    if "properties" in instance:
        schedule.append("{\n")
        schedule += build_schedule_from_instance(instance["properties"], indent + 2)
        schedule.append(" " * indent)
        schedule.append("}")
    else:
        for name, annotation in instance.items():
            schedule.append(" " * indent)
            schedule.append(f'"{name}": ')
            if "anyOf" in annotation:
                union_types = []
                for element in annotation["anyOf"]:
                    union_types.append(element["type"])
                schedule.append(union_types)
            elif annotation["type"] == "object":
                schedule += build_schedule_from_instance(annotation, indent)
            elif annotation["type"] == "array":
                schedule.append("[")
                schedule += build_schedule_from_instance(annotation["items"], indent)
                schedule.append("]")
            elif annotation["type"] == "integer":
                schedule.append(1)
            elif annotation["type"] == "number":
                schedule.append(1.3)
            elif "enum" in annotation:
                schedule.append(annotation["enum"])
            else:
                schedule.append(None)

            schedule.append(",\n")

    return schedule
