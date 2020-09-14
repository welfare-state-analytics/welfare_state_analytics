#!/bin/bash

apt autoremove -y
apt-get clean
rm /tmp/*

rm -rf /var/lib/apt/lists/*

# apt-get remove --purge -yqq $(apt-mark showauto)
# apt-get remove --purge -yqq $BUILD_PACKAGES
