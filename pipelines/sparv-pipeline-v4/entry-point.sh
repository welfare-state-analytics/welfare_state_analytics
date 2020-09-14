#!/bin/bash

export PATH="/home/.local/bin:$PATH"

cd /work

whoami

/home/.local/bin/sparv $@
