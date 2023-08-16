Example of a chatbot with a graphdb for memory

- [Installation](#installation)
Run pip install -r requirements.txt (in a new virtualenv), and memgraph.sh to start the db
Then python main.py to start the chatbot

- [TODO](#todo)
Bunch of things. 
- Conversations where user input has multiple replies from different bots (think imageboard style threads), and we can also do things like fork & join em. 
-Represent world states as nodes, and actions as edges, and then use that for planning in a game or similar apps.
- Represent semantic knowledge as a graph, relations between users, social networks etc
- Add an index on the vector embedding propery & add an example of searching with it
- Manage trees of actions, ala Voyager, for use in planning
- Relate all the above subgraphs together, eg check semantic knowledge from some entity
in the world state

- [DB usage](#usage)
After starting memgraph, go to localhost:3000 to access the visualiser. You can run queries from there
Most useful ones:
- Print all nodes: MATCH (n) RETURN n
- Drop graph & start fresh: MATCH (n) DETACH DELETE n