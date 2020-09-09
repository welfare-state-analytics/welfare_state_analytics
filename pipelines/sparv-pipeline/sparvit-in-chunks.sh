
corpus_file="SOU-KBLAB-corpus-1945-1989.xml.zip"
chunk_pattern="sou_CHUNK_"

chunks=$(seq 1945 1989)

for chunk in ${chunks[@]}; do

    xml_files=$(unzip -Z1 ${corpus_file})

    chunk_filenames=$(IFS=$'\n' && echo "${xml_files[*]}" | grep "${chunk_pattern/CHUNK/$chunk}" )

    mkdir -p ${chunk} ${chunk}/original

    for f in ${chunk_filenames}; do
        unzip ${corpus_file} ${f} -d ${chunk}/original
    done

    cp Makefile ${chunk}/

    cd ${chunk}

    sparvit export

    cd ..

    zip -rjD ${chunk}.xml.zip ${chunk}/export.original/*.xml

    cp ${chunk}/warnings.log ./${chunk}.warnings.log
    gzip ${chunk}.warnings.log

    rm -rf ${chunk}

done

