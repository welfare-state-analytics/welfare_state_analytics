
### Download package contents (JSON)

```bash
nohup westac/kblab/python download_tag_content_json.py xyz xyz_content_json.zip &> run.log
```

### Copy files between ZIP archives

```bash
python westac/kblab/copy_zip_archive.py xyz_content_json.zip xyz_1945-1989_content_json.zip 1945 1989
```

### Extract and store text

```bash
python westac/kblab/extract_json_text.py --source-filename xyz_content_json.zip --target-filename xyz_corpus.zip
```

### Vectorize corpus using CLI

```bash
nohup python scripts/vectorize_corpus.py --remove-accents --only-alphanumeric --to-lower --min-length 2 \
     --no-keep-numerals --no-keep-symbols \
     --meta-field "year:_:1" --meta-field "extra_id:_:-3" --meta-field "serial_id:_:-2" \
     data/propositioner/riksdagens_propositioner_1945-1989_corpus.zip \
     ./data/propositioner &> ./data/propositioner/run_20200828.log &
```
