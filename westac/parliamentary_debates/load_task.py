from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

from penelope.pipeline import (
    ContentType,
    CountTokensMixIn,
    DefaultResolveMixIn,
    DocumentPayload,
    ITask,
    Token2Id,
    checkpoint,
)
from penelope.pipeline import CorpusSerializeOpts
from penelope.corpus import TextReaderOpts

"""
This module implements a task that loads annotated ParlaCLARIN parliamentary debates into penelope pipeline.

Each debate protocol is stored in a ZIP that contains:

1. Annotated CSV files for each individual speech
2. A document index that for included speeches.

The Task should:
1. Generate a stream of DocumentPayload of a configurable granularity

Sample CSV (first few lines):
    text	lemma	pos	xpos
    Herr	herr	NN	NN.UTR.SIN.IND.NOM
    talman	talman	NN	NN.UTR.SIN.IND.NOM
    !	!	MID	MID
    Innan	innan	SN	SN
    jag	jag	PN	PN.UTR.SIN.DEF.SUB
    går	gå	VB	VB.PRS.AKT
    in	in	PL	PL
    på	på	PP	PP
    de	den	DT	DT.UTR+NEU.PLU.DEF
    delar	del	NN	NN.UTR.PLU.IND.NOM
    av	av	PP	PP

Sample document index:
	speech_id	speaker	speech_date	speech_index	document_name	filename	num_tokens	num_words
0	i-e0b362d758b46a2f-0	hans_hagnell_f03a00	1964-05-25	1	1	prot-1964--ak--27@1.csv	105	105
1	i-e0b362d758b46a2f-1	hans_hagnell_f03a00	1964-05-25	2	1	prot-1964--ak--27@2.csv	302	302
2	i-e0b362d758b46a2f-2	hans_hagnell_f03a00	1964-05-25	3	1	prot-1964--ak--27@3.csv	237	237
3	i-e0b362d758b46a2f-3	hans_hagnell_f03a00	1964-05-25	4	1	prot-1964--ak--27@4.csv	137	137
4	i-b995e9f8933502ad-0	unknown	1964-05-25	5	1	prot-1964--ak--27@5.csv	396	396


Reminder:
 - Sample data exists in romulus (/data, mount point /mnt/wsl/data), accessable from viavulcan via  wsl-mounted on vulcan/data

\\wsl$\\romulus\\data\\annotated


"""

ProtocolDict = dict


@dataclass
class LoadToTaggedFrame(CountTokensMixIn, DefaultResolveMixIn, ITask):
    """Loads parliamentary debates protocols stored as Sparv CSV into pipeline """

    # TODO: Add relevant filters!
    source_folder: str = None
    attributes: List[str] = None
    attribute_value_filters: Dict[str, Any] = None
    file_pattern: str = "*.zip"
    token2id: Token2Id = None
    reader_opts: TextReaderOpts = None
    serializer_opts: CorpusSerializeOpts = None

    def setup(self) -> ITask:
        self.pipeline.put("tagged_attributes", self.attributes)
        return self

    def __post_init__(self):
        self.in_content_type = [ContentType.TEXT, ContentType.TOKENS]
        self.out_content_type = ContentType.TAGGED_FRAME

    def outstream(self) -> Iterable[DocumentPayload]:

        # TODO: Build Vocabulary
        for checkpoint_data in self.protocol_stream():
            # TODO: Apply filters
            # TODO: Append document index, fix document_id
            #self.pipeline.payload.effective_document_index += checkpoint_data.document_index
            for payload in checkpoint_data.payload_stream:
                yield payload


    def protocol_stream(self) -> Iterable[checkpoint.CheckpointData]:
        for path in Path(self.source_folder).rglob(self.file_pattern):
            checkpoint_data: checkpoint.CheckpointData = checkpoint.load_checkpoint(path, options=self.serializer_opts)
            yield checkpoint_data



# re_protocol_header = \
#     r"([\w\d_\t])+" \
#     r"\n# protocol.date = ([\d-]+)" \
#     r"\n# protocol.name = (prot-[\w\d-]+)"

# re_speech_header = \
#     r"\n# speech.speaker = ([\w\d\_]+)" \
#     r"\n# speech.speech_date = ([\d-]+)" \
#     r"\n# speech.speech_id = ([\w\d\-]+)" \
#     r"\n# speech.speech_index = (\d+)"

# re_speech_index = r"\n# speech.speech_index = (\d+)"

    #     # Files are stored as compressed CSV files
    #     self.pipeline.payload.effective_document_index = pd.DataFrame(
    #         data={
    #             'document_name': [],
    #             'filename': [],
    #             'document_id': [],
    #             'year': [],
    #             'date': [],
    #             'speaker': [],
    #             'speaker_id': [],
    #             'speaker_index': [],
    #         }
    #     )
    #     self.token2id = Token2Id()

    #     protocol_stream = load_protocol_stream(filename)
    #     for protocol in protocol_stream:
    #         for speech in protocol.speeces:
    #             tagged_frame: TaggedFrame = to_tagged_frame(protocol, speech)
    #             payload = payload.update(self.out_content_type, tagged_frame)
    #             self.add_document(speech)
    #             self.register_token_counts(payload)
    #             yield payload

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
