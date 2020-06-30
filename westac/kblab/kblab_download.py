import os
import zipfile
import fnmatch

from kblab import Archive
from westac.common import utility

logger = utility.setup_logger()

def kblab_connect():

    kb_archive = Archive('https://betalab.kb.se/', auth=(os.environ['KBLAB_USER'], os.environ['KBLAB_PASSWD']))

    return kb_archive

def kblab_find_packages(kb_archive, query, max_count):

    for package_id in kb_archive.search(query, max=max_count):

        package = kb_archive.get(package_id)

        if 'content.json' not in package:
            logger.warning("skipping {} (has no content.json)".format(package_id))
            continue

        yield package_id, package

def kblab_download_package_items(package, includes=None, excludes=None):

    for filename in sorted(package.list()):

        if any(fnmatch.fnmatch(filename, pattern) for pattern in (excludes or [])):
            # print("skipping: {}".format(filename))
            continue

        if includes is None or any(fnmatch.fnmatch(filename, pattern) for pattern in includes):

            content = package.get_raw(filename).read()

            yield filename, content

def kblab_download_query_to_zip(query, max_count, target_filename, includes=None, excludes=None):

    kb_archive = kblab_connect()

    with zipfile.ZipFile(target_filename, "w") as target_archive:

        for package_id, package in kblab_find_packages(kb_archive, query, max_count):

            for filename, content in kblab_download_package_items(package, includes=includes, excludes=excludes):

                target_archive.writestr(os.path.join(package_id, filename), content, zipfile.ZIP_DEFLATED)

            logger.info("Package %s added..." % package_id)
