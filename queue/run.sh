docker run \
    -v /codabench/storage:/codabench \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -d \
    --env-file queue.env \
    --name compute_worker \
    --restart unless-stopped \
    --log-opt max-size=50m \
    --log-opt max-file=3 \
    codalab/competitions-v2-compute-worker:latest
