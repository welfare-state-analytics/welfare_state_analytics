
### Build

```bash
docker build -t sparv-pipeline:latest .
```

### Annotate

See [Preparing Corpus](https://spraakbanken.gu.se/en/tools/sparv/pipeline/installation).

1. Create a new project folder for your
1. Copy template Makefile to project folder
   - Name corpus (default `corpusname`)
   - Name document sub-folder (default `original`)
   - Specify the annotations types you want
   - Specify xml (root) tag
1. Prepare and store the corpus in a sub-folder
   - Sub-folder must be named `original` or as specified in Makefile
   - Each document must be a separate XML file
   - Each document must have a root-tag()).
1. Run sparvit.sh from witihn the sub-folder

### Example Makefile (minimal)

```txt
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


### Build Språkbanken's Sparv Pipeline

See [README](https://github.com/spraakbanken/sparv-pipeline) and [Install Instructions](https://spraakbanken.gu.se/en/tools/sparv/pipeline/installation)

Note: Set the following enviroment variables

```bash
export SPARV_MAKEFILES=`pwd`/makefiles
export SPARV_PIPELINE_PATH=`pwd`
```