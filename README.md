# model-free-rl

## System setup
Ubuntu 23
Python 3.11

You might need to check for deadsnakes PPA for getting pyhton3.11 if you are on ubuntu 24+

```sh
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11
```

You might also need to install the GPU drivers. We are using 545, but you might need to get the proper version for your system.

```sh
sudo apt install nvidia-driver-545
sudo apt install nvidia-cudnn
```

Not sure nvidia-cudnn is necessary

## Create and activate The Enviroment

```sh
    python -m venv mfrl
    source mfrl/bin/activate
    pip install -r requirements.txt
```

Make easier to activate by adding an alias in the `<shell>rc` file `alias mfrl="source <folder>/mfrl/bin/activate"`

