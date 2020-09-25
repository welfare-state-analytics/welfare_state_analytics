
## How to prepare Riksdagens Protokoll for TA

### Download package content from KB-LAB API (JSON)

The Python script `extract_json_text.py` downloads `content.json` and `meta.json` for all packages having tag "protocoll" (query `{ "tags": "protokoll" }`).

```bash
cd source/westac_data
pipenv shell
cd src/kb_labb
nohup python download_protocol_content_json.py >& run.log &
```

The result is stored in a Zip archive.

### Extract text from JSON

Use the script `extract_json_text.py` to extract the text from the JSON files.

```bash
python extract_json_text.py --source-filename ~/tmp/riksdagens_protokoll_content.zip --target-filename ~/tmp/riksdagens_protokoll_content_corpus.zip
```

The resulting Zip file contains text files named as `prot_yyyyyy__NN.txt` for each protocol.

### Prepare text files for Sparv

The Sparv pipeline requires that the individual document are stored as (individual) XML files. The shell script `sparvit-to-xml` can be used to add a root tag to all text files in a Zip archive. The resulting XML files iare stored as a new Zip archive.

```bash
 sparvit-to-xml --input riksdagens_protokoll_content_corpus.zip --output riksdagens_protokoll_content_corpus_xml.zip
 ```

