export BROKER_URL= # PUT BROKER_URL from https://www.codabench.org/queues/ here

docker run \
    -v $PWD/storage:/codabench \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -d \
    --env BROKER_URL=$BROKER_URL \
    --env HOST_DIRECTORY=$PWD/storage \
    --name compute_worker \
    --restart unless-stopped \
    --log-opt max-size=50m \
    --log-opt max-file=3 \
    codalab/competitions-v2-compute-worker:nvidia
