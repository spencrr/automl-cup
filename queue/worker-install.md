# Install Docker

```sh
sudo apt-get remove -y docker docker-engine docker.io containerd runc
sudo apt-get update -y
sudo apt-get install -y \
    ca-certificates \
    curl \
    gnupg
sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg
echo \
    "deb [arch="$(dpkg --print-architecture)" signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
    sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update -y
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

## Add Docker group

```sh
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

## Test Docker

```sh
docker run hello-world
```

# NVIDIA driver installation

```sh
sudo apt-get install -y ubuntu-drivers-common
sudo ubuntu-drivers install --gpgpu nvidia:525-server
sudo apt-get install -y nvidia-utils-525-server

sudo apt-get install -y nvidia-container-runtime
```

## Test NVIDIA drivers from Docker

```sh
docker run -it --rm --gpus all ubuntu nvidia-smi
```
