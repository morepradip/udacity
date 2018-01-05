
/* This sql script is used to create data base from csv files */
/* create nodes table */

.mode csv
.import nodes.csv nodes
ALTER TABLE nodes RENAME TO old_nodes;

CREATE TABLE nodes (
    id INTEGER PRIMARY KEY NOT NULL,
    lat REAL,
    lon REAL,
    user TEXT,
    uid INTEGER,
    version INTEGER,
    changeset INTEGER,
    timestamp TEXT
);

INSERT INTO nodes SELECT * FROM old_nodes;
DROP TABLE old_nodes;

/* create nodes_tags table */

CREATE TABLE nodes_tags (
    id INTEGER,
    key TEXT,
    value TEXT,
    type TEXT,
    FOREIGN KEY (id) REFERENCES nodes(id)
);

.import nodes_tags.csv nodes_tags
DELETE FROM nodes_tags WHERE id = 'id';


/* create ways table*/

.import ways.csv ways
ALTER TABLE ways RENAME TO old_ways;

CREATE TABLE ways (
    id INTEGER PRIMARY KEY NOT NULL,
    user TEXT,
    uid INTEGER,
    version TEXT,
    changeset INTEGER,
    timestamp TEXT
);

INSERT INTO ways SELECT * FROM old_ways;
DROP TABLE old_ways;

/* create ways_tags table*/

CREATE TABLE ways_tags (
    id INTEGER NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    type TEXT,
    FOREIGN KEY (id) REFERENCES ways(id)
);

.import ways_tags.csv ways_tags
DELETE FROM ways_tags WHERE id = 'id';

/* create ways_nodes table*/

CREATE TABLE ways_nodes (
    id INTEGER NOT NULL,
    node_id INTEGER NOT NULL,
    position INTEGER NOT NULL,
    FOREIGN KEY (id) REFERENCES ways(id),
    FOREIGN KEY (node_id) REFERENCES nodes(id)
);

.import ways_nodes.csv ways_nodes
DELETE FROM ways_nodes WHERE id = 'id';

/* check database*/
.tables
.mode column
.headers on
SELECT * FROM nodes LIMIT 5;
SELECT * FROM nodes_tags LIMIT 5;
SELECT * FROM ways LIMIT 5;
SELECT * FROM ways_tags LIMIT 5;
SELECT * FROM ways_nodes LIMIT 5;

SELECT count(*) FROM nodes;
SELECT count(*) FROM nodes_tags;
SELECT count(*) FROM ways;
SELECT count(*) FROM ways_tags;
SELECT count(*) FROM ways_nodes;
