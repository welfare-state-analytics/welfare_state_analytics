#!/bin/bash

adduser $LAB_USER --uid $LAB_UID --gid $LAB_GID --disabled-password --gecos '' --shell /bin/bash

adduser $LAB_USER sudo

echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

mkdir -p $HOME/work
