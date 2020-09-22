import fnmatch
import os
import zipfile

from kblab import Archive
from westac.common import utility

logger = utility.setup_logger()

# pylint: disable=logging-not-lazy, unnecessary-comprehension

def connect():
    """Connect to KB-LAB API

    Returns
    -------
    Archive
        KB-LAB API Archive object
    """
    archive = Archive('https://betalab.kb.se/', auth=(os.environ['KBLAB_USER'], os.environ['KBLAB_PASSWD']))

    return archive

def find_packages(archive, query, max_count):
    """Finds packages matching query.

    Parameters
    ----------
    archive : Archive
        KB-LAB archive
    query : Dict
        Query specification
    max_count : int
        Max number of packages to retrieve

    Yields
    -------
    Iterator[Tuple[str,Package]]
        A stream of (package-id, package) tuples
    """
    for package_id in archive.search(query, max=max_count):

        package = archive.get(package_id)

        if 'content.json' not in package:
            logger.warning("skipping {} (has no content.json)".format(package_id))
            continue

        yield package_id, package

def download_package_items(package, includes=None, excludes=None):
    """Downloads files in package `package`.

    Parameters
    ----------
    package : Package
        KB-LAB Package specification
    includes : List[str], optional
        List of file patterns to include in download, by default None
    excludes : List[str], optional
        List of file patterns to exclude in download, by default None

    Yields
    -------
    Tuple[str, bytes]
        Stream of filename and file content tuples.
    """
    for filename in sorted(package.list()):

        if any(fnmatch.fnmatch(filename, pattern) for pattern in (excludes or [])):
            # print("skipping: {}".format(filename))
            continue

        if includes is None or any(fnmatch.fnmatch(filename, pattern) for pattern in includes):

            content = package.get_raw(filename).read()

            yield filename, content

def download_query_to_zip(query, max_count, target_filename, includes=None, excludes=None, append=False):
    """Downloads file contents matching query and stores result in a Zip archive.

    Parameters
    ----------
    query : Dict
        Query specification
    max_count : int
        Max number of packages to retrieve
    target_filename : str
        Target Zip filename
    includes : List[str], optional
        List of file patterns to include in download, by default None
    excludes : List[str], optional
        List of file patterns to exclude in download, by default None
    """
    archive = connect()

    mode = "a" if append else "w"

    with zipfile.ZipFile(target_filename, mode) as target_archive:

        existing_files = { x for x in target_archive.namelist() }

        for package_id, package in find_packages(archive, query, max_count):

            if os.path.join(package_id, "content.json") in existing_files:
                logger.info("Skipped (exists): %s" % package_id)
                continue

            for filename, content in download_package_items(package, includes=includes, excludes=excludes):

                target_archive.writestr(os.path.join(package_id, filename), content, zipfile.ZIP_DEFLATED)

            logger.info("Added: %s" % package_id)
