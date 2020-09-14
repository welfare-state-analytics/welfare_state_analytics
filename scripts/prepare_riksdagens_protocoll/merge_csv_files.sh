#!/bin/bash

shopt -s extglob

data_folder="/home/roger/source/welfare-state-analytics/welfare_state_analytics/data/riksdagens_protokoll"

zip_files="$data_folder/ROD-DATA/"*.csv.zip
target_file="$data_folder"/prot-1971-2021.csv

rm -f $target_file
for f in $zip_files
do
    unzip -p $f >> $target_file
done
