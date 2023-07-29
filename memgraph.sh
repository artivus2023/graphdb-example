#!/bin/bash

CONTAINER_NAME="memgraph"

if [ "$1" == "start" ]; then
    echo "Starting Memgraph..."
    docker run -it -p 7687:7687 -p 7444:7444 -p 3000:3000 -v mg_lib:/var/lib/memgraph --name $CONTAINER_NAME memgraph/memgraph-platform
elif [ "$1" == "stop" ]; then
    echo "Stopping Memgraph..."
    docker stop $CONTAINER_NAME
else
    echo "Unknown command. Use either 'start' or 'stop'"
fi