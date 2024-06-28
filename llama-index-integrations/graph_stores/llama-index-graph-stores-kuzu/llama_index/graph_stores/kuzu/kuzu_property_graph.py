from typing import Any, List, Dict, Optional, Tuple

import kuzu
from llama_index.core.graph_stores.types import (
    PropertyGraphStore,
    Triplet,
    LabelledNode,
    Relation,
    EntityNode,
    ChunkNode,
)
from llama_index.core.vector_stores.types import VectorStoreQuery
from llama_index.core.graph_stores.utils import value_sanitize
import utils

# Threshold for max number of returned triplets
LIMIT = 10


class KuzuPropertyGraphStore(PropertyGraphStore):
    """
    Kùzu Property Graph Store.

    This class implements a Kùzu property graph store.

    Kùzu can be installed and used with this simple command:

    ```
    pip install kuzu
    ```
    """

    def __init__(
        self,
        db: str,
        overwrite: bool = False,
        entities: Optional[List[str]] = None,
        relationship_schema: Optional[List[Tuple[str, str, str]]] = None,
        sanitize_query_output: bool = True,
    ) -> None:
        if overwrite:
            utils.create_fresh_database(db)
        self.db = kuzu.Database(db)
        self.connection = kuzu.Connection(self.db)
        self.relationship_schema = relationship_schema
        self.has_structured_schema = True
        if not all((entities, relationship_schema)):
            self.has_structured_schema = False
            # When no entities are specified, we have a "loose" schema
            # In this situation, we create generic Entity to Entity nodes and their relationships
            self.relationship_schema = [("Entity", "LINKS", "Entity")]
        self.entities = (
            utils.get_list_from_literal(entities) if entities else ["Entity"]
        )
        # Always include Chunk as a node type
        self.entities.extend(["Chunk"])
        self.sanitize_query_output = sanitize_query_output
        self.structured_schema = {}
        self.init_schema()

    def init_schema(self) -> None:
        """Initialize schema if the required tables do not exist."""
        utils.create_chunk_node_table(self.connection)
        utils.create_entity_node_tables(self.connection, entities=self.entities)
        utils.create_relation_tables(
            self.connection,
            self.entities,
            relationship_schema=self.relationship_schema,
        )

    @property
    def client(self) -> kuzu.Connection:
        return self.connection

    def upsert_nodes(self, nodes: List[LabelledNode]) -> None:
        entity_list: List[EntityNode] = []
        chunk_list: List[ChunkNode] = []

        for item in nodes:
            if isinstance(item, EntityNode):
                entity_list.append(item)
            elif isinstance(item, ChunkNode):
                chunk_list.append(item)

        for chunk in chunk_list:
            upsert_chunk_node_query = """
                MERGE (c:Chunk {id: $id})
                  SET c.text = $text,
                      c.label = $label,
                      c.embedding = $embedding,
                      c.ref_doc_id = $ref_doc_id,
                      c.creation_date = date($creation_date),
                      c.last_modified_date = date($last_modified_date),
                      c.file_name = $file_name,
                      c.file_path = $file_path,
                      c.file_size = $file_size,
                      c.file_type = $file_type
            """

            self.connection.execute(
                upsert_chunk_node_query,
                parameters={
                    "id": chunk.id_,
                    "text": chunk.text.strip(),
                    "label": chunk.label,
                    "embedding": chunk.embedding,
                    "ref_doc_id": chunk.properties.get("ref_doc_id"),
                    "creation_date": chunk.properties.get("creation_date"),
                    "last_modified_date": chunk.properties.get("last_modified_date"),
                    "file_name": chunk.properties.get("file_name"),
                    "file_path": chunk.properties.get("file_path"),
                    "file_size": chunk.properties.get("file_size"),
                    "file_type": chunk.properties.get("file_type"),
                },
            )

            # Merge an implicit chunk node with the ref_doc_id
            # This is to be connected to the source chunk node via a relationship
            # TODO: Combine this query with the above query in Kùzu 0.5.0
            self.connection.execute(
                """
                MERGE (c:Chunk {id: $ref_doc_id})
                """,
                parameters={
                    "ref_doc_id": chunk.properties.get("ref_doc_id", "unknown"),
                },
            )
        for entity in entity_list:
            # Add entity node and its relationship with the source chunk
            upsert_entity_node_query = f"""
                MERGE (e:{entity.label} {{id: $id}})
                SET e.label = $label,
                    e.name = $name,
                    e.embedding = $embedding,
                    e.creation_date = date($creation_date),
                    e.last_modified_date = date($last_modified_date),
                    e.file_name = $file_name,
                    e.file_path = $file_path,
                    e.file_size = $file_size,
                    e.file_type = $file_type
                """

            self.connection.execute(
                upsert_entity_node_query,
                parameters={
                    "id": entity.name,
                    "label": entity.label,
                    "name": entity.name,
                    "embedding": entity.embedding,
                    "creation_date": entity.properties.get("creation_date"),
                    "last_modified_date": entity.properties.get("last_modified_date"),
                    "file_name": entity.properties.get("file_name"),
                    "file_path": entity.properties.get("file_path"),
                    "file_size": entity.properties.get("file_size"),
                    "file_type": entity.properties.get("file_type"),
                },
            )

    def upsert_relations(self, relations: List[Relation]) -> None:
        for rel in relations:
            if rel.label == "SOURCE":
                # This is a relationship between two chunks
                self.connection.execute(
                    """
                    MATCH (a:Chunk {id: $source_id}), (b:Chunk {id: $target_id})
                    MERGE (a)-[:LINKS_Chunk_Chunk {label: $label}]->(b)
                    """,
                    parameters={
                        "source_id": rel.source_id,
                        "target_id": rel.target_id,
                        "label": rel.label,
                    },
                )
            else:
                if self.has_structured_schema:
                    src, _, dst = utils.lookup_relation(
                        rel.label, self.relationship_schema
                    )
                else:
                    src, dst = "Entity", "Entity"

                rel_tbl_name = f"LINKS_{src}_{dst}"
                self.connection.execute(
                    f"""
                    MATCH (a:{src} {{id: $source_id}}), (b:{dst} {{id: $target_id}})
                    MERGE (a)-[:{rel_tbl_name} {{label: $label}}]->(b)
                    """,
                    parameters={
                        "source_id": rel.source_id,
                        "target_id": rel.target_id,
                        "label": rel.label,
                    },
                )
                # TODO: Combine the following two queries into one in Kùzu 0.5.0
                self.connection.execute(
                    f"""
                    MATCH (a:{src} {{id: $source_id}}), (c:Chunk {{id: $triplet_source_id}})
                    MERGE (c)-[:LINKS_Chunk_{src} {{label: "MENTIONS"}}]->(a)
                    """,
                    parameters={
                        "source_id": rel.source_id,
                        "triplet_source_id": rel.properties.get(
                            "triplet_source_id", "unknown"
                        ),
                    },
                )
                # This query can be combined with the above query in Kùzu 0.5.0
                self.connection.execute(
                    f"""
                    MATCH (b:{dst} {{id: $target_id}}), (c:Chunk {{id: $triplet_source_id}})
                    MERGE (c)-[:LINKS_Chunk_{dst} {{label: "MENTIONS"}}]->(b)
                    """,
                    parameters={
                        "target_id": rel.target_id,
                        "triplet_source_id": rel.properties.get(
                            "triplet_source_id", "unknown"
                        ),
                    },
                )

    def get(
        self,
        ids: Optional[List[str]] = None,
    ) -> List[LabelledNode]:
        """Get nodes from the property graph store."""
        cypher_statement = "MATCH (e) "

        parameters = {}
        if ids:
            cypher_statement += "WHERE e.id in $ids "
            parameters["ids"] = ids

        return_statement = "RETURN e.*"
        cypher_statement += return_statement
        result = self.structured_query(cypher_statement, param_map=parameters)
        result = result if result else []

        nodes = []
        for record in result:
            # Text indicates a chunk node
            # None on the type indicates an implicit node, likely a chunk node
            if record.get("e.text") or record.get("e.label") is None:
                properties = {
                    k: v for k, v in record.items() if k not in ["e.id", "e.text"]
                }
                text = record["e.text"] if record.get("e.text") else ""
                nodes.append(
                    ChunkNode(
                        id_=record["e.id"],
                        text=text,
                        properties=utils.remove_empty_values(properties),
                    )
                )
            else:
                properties = {
                    k: v for k, v in record.items() if k not in ["e.id", "e.name"]
                }
                nodes.append(
                    EntityNode(
                        name=record["e.name"],
                        label=record["e.label"],
                        properties=utils.remove_empty_values(properties),
                    )
                )
        return nodes

    def get_triplets(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Triplet]:
        # Construct the Cypher query
        cypher_statement = "MATCH (e)-[r]->(t) "

        params = {}
        if entity_names or relation_names or ids:
            cypher_statement += "WHERE "

        if entity_names:
            cypher_statement += "e.name in $entity_names "
            params["entity_names"] = entity_names

        if relation_names and entity_names:
            cypher_statement += f"AND "
        if relation_names:
            cypher_statement += "r.label in $relation_names "
            params[f"relation_names"] = relation_names

        if ids:
            cypher_statement += "e.id in $ids "
            params["ids"] = ids

        # Avoid returning a massive list of triplets that represent a large portion of the graph
        # This uses the LIMIT constant defined at the top of the file
        if not (entity_names or relation_names or ids):
            return_statement = f"WHERE e.label <> 'text_chunk' RETURN * LIMIT {LIMIT};"
        else:
            return_statement = f"AND e.label <> 'text_chunk' RETURN * LIMIT {LIMIT};"

        cypher_statement += return_statement

        result = self.structured_query(cypher_statement, param_map=params)
        result = result if result else []

        triples = []
        for record in result:
            if record["e"]["_label"] == "Chunk":
                continue

            src_table = record["e"]["_id"]["table"]
            dst_table = record["t"]["_id"]["table"]
            id_map = {src_table: record["e"]["id"], dst_table: record["t"]["id"]}
            source = EntityNode(
                name=record["e"]["id"],
                label=record["e"]["_label"],
                properties=utils.get_filtered_props(record["e"], ["_id", "_label"]),
            )
            target = EntityNode(
                name=record["t"]["id"],
                label=record["t"]["_label"],
                properties=utils.get_filtered_props(record["t"], ["_id", "_label"]),
            )
            rel = Relation(
                source_id=id_map.get(record["r"]["_src"]["table"], "unknown"),
                target_id=id_map.get(record["r"]["_dst"]["table"], "unknown"),
                label=record["r"]["label"],
            )
            triples.append([source, rel, target])
        return triples

    def get_rel_map(
        self,
        graph_nodes: List[LabelledNode],
        depth: int = 2,
        limit: int = 1,
        ignore_rels: Optional[List[str]] = None,
    ) -> List[Triplet]:
        triples = []

        ids = [node.id for node in graph_nodes]
        # Run recursive query)
        response = self.structured_query(
            f"""
            match (e)
            where e.id in $ids
            MATCH (e)-[rel*1..{depth} (r, n | WHERE r.label <> "MENTIONS" AND r.label <> "SOURCE") ]->(other)
            RETURN *
            LIMIT {limit};
            """,
            param_map={"ids": ids},
        )

        ignore_rels = ignore_rels or []
        for record in response:
            for item in record["rel"]["_rels"]:
                if item["label"] in ignore_rels:
                    continue

                src_table = item["_src"]["table"]
                dst_table = item["_src"]["table"]
                id_map = {
                    src_table: record["e"]["_id"],
                    dst_table: record["other"]["id"],
                }
                source = EntityNode(
                    name=record["e"]["name"],
                    label=record["e"]["_label"],
                    properties=utils.get_filtered_props(
                        record["e"], ["_id", "name", "_label"]
                    ),
                )
                target = EntityNode(
                    name=record["other"]["name"],
                    label=record["other"]["_label"],
                    properties=utils.get_filtered_props(
                        record["e"], ["_id", "name", "_label"]
                    ),
                )
                rel = Relation(
                    source_id=id_map.get(item["_src"]["table"], "unknown"),
                    target_id=id_map.get(item["_dst"]["table"], "unknown"),
                    label=item["label"],
                )
                triples.append([source, rel, target])

        return triples

    def delete(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        properties: Optional[dict] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Delete nodes and relationships from the property graph store."""
        if entity_names:
            self.structured_query(
                "MATCH (n) WHERE n.name IN $entity_names DETACH DELETE n",
                param_map={"entity_names": entity_names},
            )

        if ids:
            self.structured_query(
                "MATCH (n) WHERE n.id IN $ids DETACH DELETE n",
                param_map={"ids": ids},
            )

        if relation_names:
            for rel in relation_names:
                src, _, dst = utils.lookup_relation(rel, self.relationship_schema)
                self.structured_query(
                    f"""
                    MATCH (:{src})-[r {{label: $label}}]->(:{dst})
                    DELETE r
                    """,
                    param_map={"label": rel},
                )

        if properties:
            assert isinstance(
                properties, dict
            ), "`properties` should be a key-value mapping."
            cypher = "MATCH (e) WHERE "
            prop_list = []
            params = {}
            for i, prop in enumerate(properties):
                prop_list.append(f"e.`{prop}` = $property_{i}")
                params[f"property_{i}"] = properties[prop]
            cypher += " AND ".join(prop_list)
            self.structured_query(cypher + " DETACH DELETE e", param_map=params)

    def structured_query(
        self, query: str, param_map: Optional[Dict[str, Any]] = None
    ) -> Any:
        response = self.connection.execute(query, parameters=param_map)
        column_names = response.get_column_names()
        result = []
        while response.has_next():
            row = response.get_next()
            result.append(dict(zip(column_names, row)))

        if self.sanitize_query_output:
            return value_sanitize(result)

        return result

    def vector_query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> Tuple[List[LabelledNode], List[float]]:
        raise NotImplementedError(
            "Vector query is not currently implemented for KuzuPropertyGraphStore."
        )

    def get_schema(self) -> Any:
        """
        Returns a structured schema of the property graph store.

        The schema contains `node_props`, `rel_props`, and `relationships` keys and
        the associated metadata.
        Example output:
        {
            'node_props': {'Chunk': [{'property': 'id', 'type': 'STRING'},
                                    {'property': 'text', 'type': 'STRING'},
                                    {'property': 'label', 'type': 'STRING'},
                                    {'property': 'embedding', 'type': 'DOUBLE'},
                                    {'property': 'properties', 'type': 'STRING'},
                                    {'property': 'ref_doc_id', 'type': 'STRING'}],
                            'Entity': [{'property': 'id', 'type': 'STRING'},
                                    {'property': 'name', 'type': 'STRING'},
                                    {'property': 'label', 'type': 'STRING'},
                                    {'property': 'embedding', 'type': 'DOUBLE'},
                                    {'property': 'properties', 'type': 'STRING'}]},
            'rel_props': {'SOURCE': [{'property': 'label', 'type': 'STRING'}]},
            'relationships': [{'end': 'Chunk', 'start': 'Chunk', 'type': 'SOURCE'}]
        }
        """
        current_table_schema = {"node_props": {}, "rel_props": {}, "relationships": []}
        node_tables = self.connection._get_node_table_names()
        for table_name in node_tables:
            node_props = self.connection._get_node_property_names(table_name)
            current_table_schema["node_props"][table_name] = []
            for prop, attr in node_props.items():
                schema = {}
                schema["property"] = prop
                schema["type"] = attr["type"]
                current_table_schema["node_props"][table_name].append(schema)

        rel_tables = self.connection._get_rel_table_names()
        for i, table in enumerate(rel_tables):
            table_name = table["name"]
            prop_values = self.connection.execute(
                f"MATCH ()-[r:{table_name}]->() RETURN distinct r.label AS label;"
            )
            while prop_values.has_next():
                rel_label = prop_values.get_next()[0]
                src, dst = rel_tables[i]["src"], rel_tables[i]["dst"]
                current_table_schema["relationships"].append(
                    {"start": src, "type": rel_label, "end": dst}
                )
                current_table_schema["rel_props"][rel_label] = []
                table_details = self.connection.execute(
                    f"CALL TABLE_INFO('{table_name}') RETURN *;"
                )
                while table_details.has_next():
                    props = table_details.get_next()
                    rel_props = {}
                    rel_props["property"] = props[1]
                    rel_props["type"] = props[2]
                    current_table_schema["rel_props"][rel_label].append(rel_props)

        self.structured_schema = current_table_schema

        return self.structured_schema
