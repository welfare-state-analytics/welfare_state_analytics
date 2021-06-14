from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import pandas as pd

from penelope.corpus import TextReaderOpts, Token2Id
from penelope.corpus.document_index import DocumentIndex
from penelope.pipeline import (
    CheckpointOpts,
    CheckpointData,
    ContentType,
    CountTaggedTokensMixIn,
    DefaultResolveMixIn,
    DocumentPayload,
    ITask,
    TaggedFrame,
)
from penelope.utility import strip_extensions, strip_path_and_extension, PoS_Tag_Scheme

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

# FIXME: #141 Remove when Parla-CLARIN source data has been re-processed
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


def _create_empty_merged_document_index() -> pd.DataFrame:
    _document_index: pd.DataFrame = pd.DataFrame(
        {
            'document_name': pd.Series([], 'str'),
            'filename': pd.Series([], 'str'),
            'year': pd.Series([], 'int'),
            'document_date': pd.Series([], 'str'),
            'num_tokens': pd.Series([], 'int'),
            'num_words': pd.Series([], 'int'),
            'document_id': pd.Series([], 'int'),
        }
    ).set_index('document_name', drop=False)
    return _document_index


def _create_merged_document_info(checkpoint: CheckpointData, i: int = 0) -> dict:

    document_name: str = strip_path_and_extension(checkpoint.source_name)
    date: str = str(checkpoint.document_index.speech_date.min())
    year = int(date[:4]) if len(date) >= 4 else None

    data: dict = {
        'document_name': document_name,
        'filename': f"{document_name}.csv",
        'year': year,
        'document_date': date,
        'num_tokens': checkpoint.document_index.num_tokens.sum(),
        'num_words': checkpoint.document_index.num_words.sum(),
        'document_id': i,
    }
    return data


@dataclass
class ToTaggedFrame(CountTaggedTokensMixIn, DefaultResolveMixIn, ITask):
    """Loads parliamentary debates protocols stored as Sparv CSV into pipeline """

    source_folder: str = ""
    checkpoint_opts: Optional[CheckpointOpts] = None
    checkpoint_filter: Optional[Callable[[str], bool]] = None
    reader_opts: Optional[TextReaderOpts] = None

    # Not used or not implemented:
    attribute_value_filters: Optional[Dict[str, Any]] = None
    attributes: Optional[List[str]] = None

    file_pattern: str = "*.zip"
    show_progress: bool = False
    merge_speeches: bool = False

    def setup(self) -> ITask:
        self.pipeline.put("tagged_attributes", self.attributes)
        return self

    def __post_init__(self):
        self.in_content_type = ContentType.TAGGED_FRAME
        self.out_content_type = ContentType.TAGGED_FRAME

    def process_stream(self) -> Iterable[DocumentPayload]:

        for i, checkpoint in enumerate(
            load_checkpoints(
                self.source_folder,
                file_pattern=self.file_pattern,
                checkpoint_opts=self.checkpoint_opts,
                checkpoint_filter=self.checkpoint_filter,
                reader_opts=self.reader_opts,
                show_progress=self.show_progress,
            )
        ):
            # FIXME: Updates faulty index caused by old bug where document_index was set to constant "1"
            checkpoint.document_index = temporary_update_document_index(checkpoint.document_index)

            if self.merge_speeches:

                """Return each protocol as a single document"""

                if self.pipeline.payload.effective_document_index is None:
                    self.pipeline.payload.effective_document_index = _create_empty_merged_document_index()

                payload, document_info = self.merge_checkpoint(checkpoint, i)

                self.pipeline.payload.effective_document_index.loc[document_info['document_name']] = document_info

                payload = self.process_payload(payload)
                yield payload

            else:

                """Return each speech as a single document"""

                self.pipeline.payload.extend_document_index(checkpoint.document_index)
                for payload in checkpoint.payload_stream:
                    payload = self.process_payload(payload)
                    payload.property_bag.update(checkpoint.document_index.loc[payload.document_name].to_dict())
                    yield payload

    def merge_checkpoint(self, checkpoint: CheckpointData, i: int) -> Tuple[DocumentPayload, dict]:
        """Merges speeches in a CheckpointData into a single document.

        Args:
            checkpoint (CheckpointData): A single protocol split into speeches stored as a checkpoint

        Returns:
            DocumentPayload: returned Payload
        """

        merged_content: pd.Series = pd.concat([payload.content for payload in checkpoint.payload_stream])

        payload: DocumentPayload = DocumentPayload(
            ContentType.TAGGED_FRAME,
            content=merged_content,
            filename=checkpoint.source_name,
        )

        document_info = _create_merged_document_info(checkpoint=checkpoint, i=i)

        return payload, document_info

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


# @dataclass
# class ToIdTaggedFrame(ToTaggedFrame):
#     """Loads parliamentary debates protocols using id-word mappings
#     Resulting data frame will have columns `token_id` and `pos_id`
#     """

#     token2id: Token2Id = field(default=Token2Id())
#     lemmatize: bool = False

#     def __post_init__(self):
#         super().__post_init__()
#         self.out_content_type = ContentType.TAGGED_ID_FRAME

#     def setup(self) -> ITask:
#         super().setup()
#         self.token2id = self.token2id or Token2Id()
#         return self

#     def process_payload(self, payload: DocumentPayload) -> DocumentPayload:

#         tagged_frame: TaggedFrame = payload.content

#         pos_schema = self.pipeline.config.pipeline_payload.pos_schema
#         token_column = self.checkpoint_opts.text_column_name(self.lemmatize)
#         pos_column = self.checkpoint_opts.pos_column

#         tagged_frame: TaggedFrame = codify_tagged_frame(
#             tagged_frame,
#             token2id=self.token2id,
#             pos_schema=pos_schema,
#             token_column=token_column,
#             pos_column=pos_column,
#         )

#         return payload.update(ContentType.TAGGED_ID_FRAME, tagged_frame)


# def codify_tagged_frame(
#     tagged_frame: pd.DataFrame,
#     token2id: Token2Id,
#     pos_schema: PoS_Tag_Scheme,
#     token_column: str,
#     pos_column: str,
# ) -> pd.DataFrame:

#     token2id.ingest(tagged_frame[token_column])  # type: ignore

#     tagged_frame = tagged_frame.assign(
#         token_id=tagged_frame[token_column].map(token2id),
#         pos_id=tagged_frame[pos_column].map(pos_schema.pos_to_id),
#     )

#     return tagged_frame
