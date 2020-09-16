import sys
import os
import pandas as pd
import zipfile

root_folder = os.path.join(os.getcwd().split('welfare_state_analytics')[0], 'welfare_state_analytics')

sys.path = list(set(sys.path + [ root_folder ]))

from text_analytic_tools.common.gensim_utility import CompressedFileReader

def load_index(filename):
    """Loads document index as a Pandas dataframe.
       The CSV file is a merge of all datasets CSV files downloaded from RÖD using merge_prot_csv.sh
       The CSV file has meen manually cleaned: \"\"\" => \"\" and invalid char exists in file (done manually)

    Parameters
    ----------
    filename : str
        Name of (cleaned) output from merge_prot_csv.sh

    Returns
    -------
    DataFrame
        Document index
    """
    meta_columns = [
        "hangar_id", "dok_id", "rm", "beteckning", "doktyp", "typ", "subtyp", "tempbeteckning", "organ",
        "mottagare", "nummer", "datum", "systemdatum", "titel", "subtitel", "status", "relaterat_id"
    ]
    #
    df = pd.read_csv(filename, header=None, sep=',', quotechar='"')
    df.columns = meta_columns
    df = df.set_index('dok_id')
    # Update faulty value: G209106	1978/79	106	prot	prot	prot				===>107<===
    #df.set_value('G209106', 'nummer', 106)
    return df

def rename_text_corpus_files(document_index, source_filename, target_filename):
    """Renames text files downloaded from RÖD in accordance to names of documents downloaded from KB-LABB.
        The files is renamed to "prot_YYYYyy_nn.txt" e.g. "prot_198990__142.txt"

    Parameters
    ----------
    document_index : DataFrame
        Document index
    source_filename : str
        Source corpus filename
    target_filename : str
        Taret corpus filename
    """
    def get_rm_tag(rm):
        """Returns 'riksmötet' as YYYYyy if rm is yyyy/yyyy else rm
        """
        rm_parts = rm.split("/")
        if len(rm_parts) == 1:
            return rm
        return rm_parts[0] + rm_parts[1][-2:]

    reader = CompressedFileReader(source_filename)

    with zipfile.ZipFile(target_filename, "w") as of:
        for document_name, content in reader:
            doc_id = document_name.split(".")[0]
            meta = document_index.loc[doc_id.upper()].to_dict()
            target_name = f'prot_{get_rm_tag(meta["rm"])}__{meta["beteckning"]}.txt'
            of.writestr(target_name, content, zipfile.ZIP_DEFLATED )

    print("Done!")

if __name__ == "__main__":

    data_folder = os.path.join(root_folder, 'data/riksdagens_protokoll')
    source_corpus_filename = os.path.join(data_folder, 'prot-1971-2021.text.zip')
    target_corpus_filename = os.path.join(data_folder, 'prot-1971-2021.corpus.txt.zip')
    index_filename = os.path.join(data_folder, 'prot-1971-2021.csv')
    excel_index_filename = os.path.join(data_folder, 'prot-1971-2021.index.xlsx')

    document_index = load_index(index_filename)

    document_index.to_excel(excel_index_filename)

    rename_text_corpus_files(document_index, source_corpus_filename, target_corpus_filename)
