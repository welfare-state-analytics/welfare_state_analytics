

### Annotate

See [Preparing Corpus](https://spraakbanken.gu.se/en/tools/sparv/pipeline/installation).

Note that the source corpus files _must_ be XML files. At the very minimum, if the files are plain text files, you need to open each file and add a root tag that encloses the text content.

#### Configure project

Create a new project folder for your annotation project. Copy the template `Makefile` and set project specific configuration elements. Specify name of corpus (default `corpusname`), name of corpus source folder (default `original`), the XML root tag (default `<text>`) and the kind of annotations to run.

```bash
% mkdir riksdagens_protokoll
% cd riksdagens_protokoll
% cp "path-to-makefile"/Makefile .
```

#### Prepare corpus

The corpus files must be stored as individual XML files in a separate sub-folder. The default name of the folder is `original`but this can be changes in the Makefile.

Script `sparvit-to-xml.sh` adds a root tag `<text>` to all files in an archive. The name of the root tag can also be specified in the Makefile.

```bash
% mkdir original
% cd original
% sparvit-to-xml.sh --file=xyz.zip
```

#### Run annotation

Run `sparvit.sh export` to convert at plain text corpus to XML files.

### Example Makefile (minimal)

```Makefile
include $(SPARV_MAKEFILES)/Makefile.config

corpus = corpusname
original_dir = original

vrt_columns_annotations = word pos msd baseform lemgram sense compwf complemgram ref dephead.ref deprel
vrt_columns             = word pos msd lemma    lex     sense compwf complemgram ref dephead     deprel
vrt_structs_annotations = sentence.id paragraph.n
vrt_structs             = sentence:id paragraph:n

xml_elements    = text
xml_annotations = text
xml_skip =

token_chunk = sentence
token_segmenter = better_word

sentence_chunk = paragraph
sentence_segmenter = punkt_sentence

paragraph_chunk = text
paragraph_segmenter = blanklines

include $(SPARV_MAKEFILES)/Makefile.rules
```

### Språkbanken's Sparv documentation

[Sparv, Github](https://github.com/spraakbanken/sparv-pipeline)

[Sparv, v2](https://ws.spraakbanken.gu.se/ws/sparv/v2/#settings)

[Sparv, v3](https://ws.spraakbanken.gu.se/docs/sparv)


### How to build Språkbanken's Sparv Pipeline

See [README](https://github.com/spraakbanken/sparv-pipeline) and [Install Instructions](https://spraakbanken.gu.se/en/tools/sparv/pipeline/installation)

Note: The following enviroment variables must be set!

```bash
cd "sparv-root-folder"
export SPARV_MAKEFILES=`pwd`/makefiles
export SPARV_PIPELINE_PATH=`pwd`
```

### Build Docker Image

```bash
export SPARV_MAKEFILES=`pwd`/makefiles
export SPARV_PIPELINE_PATH=`pwd`
```

```bash
docker build -t sparv-pipeline:latest .
```

The `Dockerfile` downloads all necessary files. Note that the `models` directory, containing models and downloaded files, is rather large and it might be a good idea to pre-compile the models and copy the `models` folder into the Docker image. An alternative way is to mount a host folder that contains the ready-to-use `models` folder.

