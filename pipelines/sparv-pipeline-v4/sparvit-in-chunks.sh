
corpus_file=xyz.zip

for chunk in chunks:

    mkdir -p ${chunk} ${chunk}/original

    cp Makefile ${chunk}/

    cd ${chunk}

    sparvit-to-xml.sh -i $corpus_file --filter ${chunk} -o corpus.xml.zip

    cd ./original && unzip ../corpus.xml.zip && cd ..

    sparvit.sh export

    zip -f original.export.zip original.export

    mv original.export.zip ../${chunk}-original.export.zip
    cd ..

    rm -rf ${chunk}



