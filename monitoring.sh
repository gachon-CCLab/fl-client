#!/bin/sh

# start the container stack
# (assumes the caller has permission to do this)
docker-compose -f docker-compose-monitoring.yml up -d

# wait for the service to be ready
while ! curl --fail --silent --head http://localhost:5601; do
  sleep 1
done

# open the browser window
open http://localhost:5601

# Source: (https://stackoverflow.com/a/70463577/20211370)