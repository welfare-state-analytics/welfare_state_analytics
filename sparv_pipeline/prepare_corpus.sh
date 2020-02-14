#!/bin/bash

for i in "$@"; do
    case $i in
        -f=*|--file*)
            source_archive="${i#*=}"
    esac
done

if [ ! -f "$source_archive" ]; then
    echo "usage: $0 --file=filename"
    exit 64
fi

target_archive="${source_archive%%.*}".xml.zip

archive_files=`unzip -Z1 $source_archive`
archive_files="${archive_files//$'\n'/ }"

rm -f $target_archive

for file in $archive_files; do

    xmlfile="${file%%.*}".xml

    rm -f "$file" "$xmlfile"
    unzip $source_archive "$file"

    echo "<text>"   > "$xmlfile"
    cat "$file"    >> "$xmlfile"
    echo "</text>" >> "$xmlfile"

    zip -u $target_archive "$xmlfile"

    #rm -f "$file" "$xmlfile"

done

