#!/bin/bash
# Converts a plain text file corpus into a XML file corpus by enclosing the text in
# each file with a `<text>` root tag.
#
# The input corpus must be a ZIP archive `xyz.zip` containg plain text files.
# The output corpus will be named `xyz.xml.zip`.
#

input_archive=""
output_folder=""
output_archive=""
root_tag="text"

for i in "$@"; do
    case $i in
        -i|--input)
            input_archive="$2"; shift; shift ;;
        -o|--output-folder)
            output_folder="$2"; shift; shift ;;
    esac
done

if [ ! -f "$input_archive" ]; then
    echo "usage: $0 --intput filename.zip"
    exit 64
fi

if [ "$output_folder" == "" ]; then
    # Write to ZIP instead of folder
    output_archive="${input_archive%%.*}".xml.zip
    rm -f $output_archive
else
    mkdir -p "$output_folder"
fi

archive_files=`unzip -Z1 $input_archive`
archive_files="${archive_files//$'\n'/ }"

for file in $archive_files; do

    xmlfile="${file%%.*}".xml

    (echo "<${root_tag}>" && unzip -p -qq $input_archive "$file" && echo "</${root_tag}>")  | \
        sed 's/&/\&amp;/g; s/</\&lt;/g; s/>/\&gt;/g' > "$xmlfile"

    if [ "$output_archive" != "" ]; then
        zip -q -u $output_archive "$xmlfile"
        rm -f "$xmlfile"
    else
        mv "$xmlfile" "$output_folder"
    fi
done
