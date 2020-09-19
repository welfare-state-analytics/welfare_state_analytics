import zipfile

import click

from . import utility
from .altoxml_parser import AltoXmlParser

# pylint: disable=no-value-for-parameter

def extract_documents(source_filename, pattern, line_break='\n', page_break='\n'):
    """Returns a stream of filename & text tuples

    Parameters
    ----------
    source_filename : str
        Filename of archive with ALTO-XML files
    target_filename : str
        Result filename
    pattern : str
        File pattern to use to restrict files
    line_break : str, optional
        Line delimiter, by default '\n'
    page_break : str, optional
        Page delimiter, by default '\n'

    Yields
    -------
    Iterator[Tuple[[str,str]]
        Result stream
    """
    parser = AltoXmlParser(line_break=line_break, page_break=page_break)

    with zipfile.ZipFile(source_filename, 'r') as zf:

        for package_id, filenames in utility.zip_folder_glob(zf, pattern):

            xml_contents = (zf.read(filename) for filename in filenames)

            document_tokens = parser.document(xml_contents)

            document = ' '.join(list(document_tokens))

            yield package_id, document

@click.command()
@click.option('--source-filename', default='alto-xml.zip', help='Source ALTO-XML filename.')
@click.option('--target-filename', default='corpus.zip', help='Target text corpus.')
@click.option('--pattern', default='*.xml', help='Filename filter on input files')
@click.option('--line-break', default='\n', help='Glue for line joins.')
@click.option('--page-break', default='\n', help='Glue for page joins.')
def extract_corpus(source_filename, target_filename, pattern, line_break='\n', page_break='\n'):
    """Extracts text from an ALTO-XML file.

    Parameters
    ----------
    source_filename : str
        Input ALTO-XML filename
    target_filename : str
        Result filename
    pattern : str
        File pattern to use to restrict files
    line_break : str, optional
        Line delimiter, by default '\n'
    page_break : str, optional
        Page delimiter, by default '\n'
    """
    texts = (
        ("{}.txt".format(package_id), text)
            for package_id, text in
                extract_documents(source_filename, pattern, line_break=line_break, page_break=page_break)
    )
    utility.store_to_zipfile(target_filename, texts)

if __name__ == "__main__":

    # source_filename = "/home/roger/tmp/riksdagens_protokoll.zip"
    # target_filename = "/home/roger/tmp/riksdagens_protokoll_corpus.zip"
    # pattern =  "prot_*.xml"
    # line_break='\n'
    # page_break='\n#########\n'

    #extract_corpus(source_filename, target_filename, pattern, line_break=line_break, page_break=page_break)

    extract_corpus()
