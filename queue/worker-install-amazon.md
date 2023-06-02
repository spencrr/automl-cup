# Install for Amazon Linux

```sh
sudo yum update
sudo yum install -y docker

sudo usermod -a -G docker ec2-user
id ec2-user
newgrp docker

sudo systemctl enable docker.service
sudo systemctl start docker.service
```

```sh
wget https://us.download.nvidia.com/tesla/525.105.17/NVIDIA-Linux-x86_64-525.105.17.run
chmod +x NVIDIA-Linux-x86_64-525.105.17.run
sudo yum install -y gcc kernel-devel kernel-headers
sudo ./NVIDIA-Linux-x86_64-525.105.17.run
```
