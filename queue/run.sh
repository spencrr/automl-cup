export BROKER_URL= # PUT BROKER_URL from https://www.codabench.org/queues/ here

docker run \
    -v $(pwd)/storage:/codabench \
    -v $(pwd)/datasets:/datasets \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -d \
    --env-file queue.env \
    --name compute_worker \
    --restart unless-stopped \
    --log-opt max-size=50m \
    --log-opt max-file=3 \
    automlcup2023/compute_worker
