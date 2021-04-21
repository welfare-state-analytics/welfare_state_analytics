from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List

from penelope.corpus import TextReaderOpts
from penelope.corpus.document_index import DocumentIndex
from penelope.pipeline import (
    CheckpointOpts,
    ContentType,
    CountTokensMixIn,
    DefaultResolveMixIn,
    DocumentPayload,
    ITask,
    TaggedFrame,
    Token2Id,
)
from penelope.utility.filename_utils import strip_extensions

from .checkpoint import load_checkpoints

"""
This module implements a task that loads annotated ParlaCLARIN parliamentary debates into penelope pipeline.

Each parliamentary protocol is stored in a ZIP that contains:
 - annotated CSV files for each individual speech
 - a document index that for included speeches
 - a JSON object containing checkpoint (serialization) options

The Task generates a stream of DocumentPayload of a configurable granularity

Sample CSV (first few lines):
    text	lemma	pos	xpos
    Herr	herr	NN	NN.UTR.SIN.IND.NOM
    talman	talman	NN	NN.UTR.SIN.IND.NOM
    !	!	MID	MID
...

Sample document index:
	speech_id	speaker	speech_date	speech_index	document_name	filename	num_tokens	num_words
0	i-e0b362d758b46a2f-0	hans_hagnell_f03a00	1964-05-25	1	1	prot-1964--ak--27@1.csv	105	105
1	i-e0b362d758b46a2f-1	hans_hagnell_f03a00	1964-05-25	2	1	prot-1964--ak--27@2.csv	302	302
...

Reminder:
 - Sample data exists in romulus (/data, mount point /mnt/wsl/data), accessable from viavulcan via  wsl-mounted on vulcan/data

\\wsl$\\romulus\\data\\annotated

"""

ProtocolDict = dict

def temporary_update_document_index(document_index: DocumentIndex) -> DocumentIndex:

    document_index = (
        document_index.assign(
            document_name=document_index.filename.apply(strip_extensions),
            document_id=range(0, len(document_index)),
        )
        .set_index('document_name', drop=False)
        .rename_axis('')
    )

    return document_index


@dataclass
class ToTaggedFrame(CountTokensMixIn, DefaultResolveMixIn, ITask):
    """Loads parliamentary debates protocols stored as Sparv CSV into pipeline """

    source_folder: str = None
    checkpoint_opts: CheckpointOpts = None
    checkpoint_filter: Callable[[str], bool] = None
    reader_opts: TextReaderOpts = None

    # Not used or not implemented:
    attribute_value_filters: Dict[str, Any] = None
    attributes: List[str] = None

    file_pattern: str = "*.zip"
    show_progress: bool = False

    def setup(self) -> ITask:
        self.pipeline.put("tagged_attributes", self.attributes)
        return self

    def __post_init__(self):
        self.in_content_type = ContentType.TAGGED_FRAME
        self.out_content_type = ContentType.TAGGED_FRAME

    def process_stream(self) -> Iterable[DocumentPayload]:

        for checkpoint in load_checkpoints(
            self.source_folder,
            file_pattern=self.file_pattern,
            checkpoint_opts=self.checkpoint_opts,
            checkpoint_filter=self.checkpoint_filter,
            reader_opts=self.reader_opts,
            show_progress=self.show_progress,
        ):
            # FIXME: Updates faulty index caused by old bug where document_index was set to constant "1"
            checkpoint.document_index = temporary_update_document_index(checkpoint.document_index)

            self.pipeline.payload.extend_document_index(checkpoint.document_index)
            for payload in checkpoint.payload_stream:
                payload = self.process_payload(payload)
                payload.property_bag.update(checkpoint.document_index.loc[payload.document_name].to_dict())
                yield payload

    # def register_document(self, payload: DocumentPayload, protocol: dict, speech: dict) -> DocumentPayload:
    #     """Add document to document index with computed token counts from the tagged frame"""
    #     try:
    #         token_counts = convert.tagged_frame_to_token_counts(
    #             tagged_frame=payload.content,
    #             pos_schema=self.pipeline.payload.pos_schema,
    #             pos_column=self.pipeline.payload.get('pos_column'),
    #         )
    #         self.update_document_properties(payload, **token_counts)
    #         return payload
    #     except Exception as ex:
    #         logging.exception(ex)
    #         raise


@dataclass
class ToIdTaggedFrame(ToTaggedFrame):
    """Loads parliamentary debates protocols using id-word mappings
    Resulting data frame will have columns `token_id` and `pos_id`
    """

    token2id: Token2Id = None
    lemmatize: bool = False

    def __post_init__(self):
        super().__post_init__()
        self.out_content_type = ContentType.TAGGED_ID_FRAME

    def setup(self) -> ITask:
        super().setup()
        self.token2id = self.token2id or Token2Id()
        return self

    def process_payload(self, payload: DocumentPayload) -> DocumentPayload:

        tagged_frame: TaggedFrame = payload.content

        pos_schema = self.pipeline.config.pipeline_payload.pos_schema
        token_column = self.checkpoint_opts.text_column_name(self.lemmatize)
        pos_column = self.checkpoint_opts.pos_column

        self.token2id.ingest(tagged_frame[token_column])

        tagged_frame = tagged_frame.assign(
            token_id=tagged_frame[token_column].map(self.token2id),
            pos_id=tagged_frame[pos_column].map(pos_schema.pos_to_id),
        )
        return payload.update(
            self.out_content_type,
            content=tagged_frame,
        )
