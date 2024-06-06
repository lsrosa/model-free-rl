# model-free-rl

## System setup
Ubuntu 23
Python 3.11

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

