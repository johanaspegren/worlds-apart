# modules/schemas/graph_schema.py

from typing import List, Optional, Union
from pydantic import BaseModel, Field, ConfigDict


class AttributeKV(BaseModel):
    """
    Simple key/value attribute for entities.
    Used for things like: founded: 1998
    """
    model_config = ConfigDict(extra="forbid")

    key: str
    value: Optional[Union[str, float, int, bool]] = None


class Entity(BaseModel):
    """
    Real-world thing: Person, Company, City, Animal, Industry, etc.
    `type` will be used directly as a Neo4j label.
    """
    model_config = ConfigDict(extra="forbid")

    id: str
    name: str
    type: str  # e.g. "Person", "Company", "Animal", "City", "Industry"
    aliases: List[str] = Field(default_factory=list)
    source_spans: List[str] = Field(default_factory=list)
    attributes: List[AttributeKV] = Field(default_factory=list)


class Relation(BaseModel):
    """
    Semantic edge between entities.
    """
    model_config = ConfigDict(extra="forbid")

    source_id: str
    type: str  # e.g. OWNS, HEADQUARTERED_IN, SERVES, LOCATED_IN, etc.
    target_id: str
    source_span: Optional[str] = None
    confidence: Optional[float] = None


class EntityList(BaseModel):
    """
    Wrapper used as structured output for entity extraction.
    """
    model_config = ConfigDict(extra="forbid")

    entities: List[Entity]


class RelationList(BaseModel):
    """
    Wrapper used as structured output for relation extraction.
    """
    model_config = ConfigDict(extra="forbid")

    relations: List[Relation]


class GraphResult(BaseModel):
    """
    Final graph passed around inside the app and into Neo4j.
    """
    model_config = ConfigDict(extra="forbid")

    entities: List[Entity]
    relations: List[Relation]
