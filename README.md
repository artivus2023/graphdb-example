# Example of a Chatbot with a GraphDB for Memory

## Table of Contents
1. [Installation](#installation)
2. [TODO](#todo)
3. [DB Usage](#usage)

## Installation
To install, run `pip install -r requirements.txt` in a new virtual environment, and `memgraph.sh` to start the database. Then, run `python main.py` to start the chatbot.

## TODO
There are several tasks to be finished:

- Implement conversations where user input has multiple replies from different bots (think imageboard style threads). We can also do things like fork & join them.
- Represent world states as nodes, and actions as edges. This can be used for planning in a game or similar applications.
- Represent semantic knowledge as a graph, including relations between users, social networks etc.
- Add an index on the vector embedding property and provide an example of searching with it.
- Manage trees of actions, similar to Voyager, for use in planning.
- Relate all the above subgraphs together. For example, check semantic knowledge from some entity in the world state.

## DB Usage
After starting Memgraph, go to `localhost:3000` to access the visualiser. You can run queries from there. Here's probably the most useful ones::

- Print all nodes: `MATCH (n) RETURN n`
- Drop graph & start fresh: `MATCH (n) DETACH DELETE n`