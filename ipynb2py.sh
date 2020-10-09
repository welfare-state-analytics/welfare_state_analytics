#!/bin/bash
remove_pairing=

for ipynb_path in `find ./notebooks/scrachpads -name "*.ipynb" -type f \( ! -name "*checkpoint*" \) -print`; do

    ipynb_filename="${ipynb_path##*/}"
    #ipynb_basename="${ipynb_filename%.*}"
    ipynb_basepath="${ipynb_path%.*}"
    py_filepath=${ipynb_basepath}.py

    #poetry run jupytext --test -x $ipynb_path --to py:percent
    #poetry run jupytext --test-strict -x --update $ipynb_path --to py:percent

    if [ "$remove_pairing" == "yes" ]; then
        echo "info: unpairing $ipynb_path..."
        poetry run jupytext --update-metadata '{"jupytext": null}' $ipynb_path &> /dev/null
        if [ -f $py_filepath ]; then
            rm $py_filepath
        fi
        continue
    fi

    if [ ! -f $py_filepath ]; then
        echo "info: pairing $ipynb_path with formats ipynb,py..."
        poetry run jupytext --quiet --set-formats ipynb,py:percent $ipynb_path
    fi

    # update the input cells in the .ipynb file and preserve outputs and metadata
    # poetry run jupytext --update --to notebook notebook.py
    # Update all paired representations of notebook.ipynb
    poetry run jupytext --quiet --sync $ipynb_path

done;


# SAVEIFS=$IFS
# IFS=$(echo -en "\n\b")

# for file in `find . -name "*.md" -print | egrep -v README.md`; do
# echo $file
# filename="${file%.*}"
# jupytext --to notebook --output "$filename.ipynb" "$filename.md"
# touch "$filename.md"
# python - <<EOF
# import json
# contents = json.loads(open("$filename.ipynb").read())
# contents["metadata"]["jupytext"] = {"formats": "ipynb,md"}
# new_content = json.dumps(contents, indent=4, sort_keys=True)
# open("$filename.ipynb","w").write(new_content)
# EOF
# rm "$filename.py"
# jupytext --to py --output $filename.py $filename.ipynb
# jupytext --test -x $filename.ipynb --to py
# #jupytext --to py --output $filename.py $filename.ipynb
# #jupytext --test -x $filename.ipynb --to py
# done;
# #jupytext --to markdown --output /results/simple-nb.md /jupyter-git/simple-nb.ipynb

# IFS=$SAVEIFS