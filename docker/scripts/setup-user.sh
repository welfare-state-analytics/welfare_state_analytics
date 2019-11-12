#!/bin/bash

adduser $PUSER --uid $PUID --gid $PGID --disabled-password --gecos '' --shell /bin/bash

adduser $PUSER sudo

echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

mkdir -p $HOME/work
