import os
import zipfile
import utility

from altoxml_parser import AltoXmlParser

def extract_documents(source_filename, pattern, line_break='\n', page_break='\n'):

    parser = AltoXmlParser(line_break=line_break, page_break=page_break)

    with zipfile.ZipFile(source_filename, 'r') as zf:

        for package_id, filenames in utility.zip_folder_glob(zf, pattern):

            xml_contents = (zf.read(filename) for filename in filenames)

            document_tokens = parser.document(xml_contents)

            document = ' '.join([ x for x in document_tokens ])

            yield package_id, document

def extract_corpus(source_filename, target_filename, pattern, line_break='\n', page_break='\n'):

    texts = (
        ("{}.txt".format(package_id), text)
            for package_id, text in
                extract_documents(source_filename, pattern, line_break=line_break, page_break=page_break)
    )
    utility.store_to_zipfile(target_filename, texts)

source_filename = "/home/roger/tmp/riksdagens_protokoll.zip"
target_filename = "/home/roger/tmp/riksdagens_protokoll_corpus.zip"
pattern =  "prot_*.xml"
line_break='\n'
page_break='\n#########\n'

extract_corpus(source_filename, target_filename, pattern, line_break=line_break, page_break=page_break)