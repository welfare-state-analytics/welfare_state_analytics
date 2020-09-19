import fnmatch
import os
import zipfile


def get_unique_folder_names(filenames):
    """Returns unique folder names in archive

    Parameters
    ----------
    filenames : str ist

    Returns
    -------
    str list
        Unique filenames sorted in ascending order
    """
    folders = set(os.path.dirname(f) for f in filenames)
    return sorted([ f for f in folders if f != '' ])

def filter_filenames(filenames, pattern):
    """Returns a fitered list of filenames that matches `pattern`

    Parameters
    ----------
    filenames : str list
    pattern : str

    Returns
    -------
    str list
        sorted list of filenames matching pattern
    """
    filenames = sorted(fnmatch.filter(filenames, pattern))

    return filenames


    # for filename, content in kblab_download_package_items(package, excludes):

    #     m = re.match(r'(\w+)_(\d{4})__(\d+)\-(\d+)\.xml', filename)
    #     if m is not None:

    #         year, _, item_id = m.groups()

    #         if (previous_id + 1) != int(item_id):
    #             print("warning: page(s) with no XML found {} expected {}".format(int(item_id), item_id + 1))

    #         page_content = xmltodict.parse(content)
    #         assert isinstance(page_content, dict) and 'alto' in page_content
    #         package_documents.append(page_content)

    #         previous_id = int(item_id)

    #     target_archive.writestr(os.path.join(package_id, filename), content, zipfile.ZIP_DEFLATED)

    # return package_documents

def store_to_zipfile(target_filename, filename_document_iter):

    with zipfile.ZipFile(target_filename, 'w') as zf:

        for filename, document in filename_document_iter:

            zf.writestr(filename, document, zipfile.ZIP_DEFLATED)

def zip_folder_glob(zf, pattern="*.xml"):
    """Returns filenames that matches `pattern` for each folder in the zip file.

    Parameters
    ----------
    zf : ZipFile
        The zip file.
    pattern : str, optional
        Filename pattern, by default "*.xml"

    Yields
    -------
    (str, str : list)
        Enumerable of sorted pairs (folder name, list of matching filenames in folder)
    """
    filenames = zf.namelist()

    folders = get_unique_folder_names(filenames)

    for folder in folders:

        matching_filenames = sorted(fnmatch.filter(filenames, os.path.join(folder, pattern)))

        if len(matching_filenames) > 1:
            yield folder, matching_filenames
