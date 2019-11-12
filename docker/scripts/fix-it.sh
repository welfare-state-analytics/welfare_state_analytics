#!/bin/bash

# Part of this code is from the jupyterLab Project
# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

function set_permissions()
{
    user_id=$1
    group_id=$2
    folder=$3

    chown -R ${user_id}:${group_id} ${folder}
    find "$folder" ! \( -group $group_id -a -perm -g+rwX \) -exec chgrp $group_id {} \; -exec chmod g+rwX {} \;
    find "$folder" \( -type d -a ! -perm -6000 \) -exec chmod +6000 {} \;
}

set_permissions ${PUID} ${PGID} ${HOME}
