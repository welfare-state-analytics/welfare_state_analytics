#-*- coding: utf-8 -*-

import re
import fileinput
import zipfile

class DocumentSplitter:

    def __init__(self, archive_name):
        self.zf = zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED)
        self.buffer = []
        self.id = None

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_value, traceback):

        self.zf.close()

    def flush(self):

        if len(self.buffer) == 0 or self.id is None:
            return

        print('flushing: {}'.format(self.id))
        self.zf.writestr("{0}.xml".format(self.id), '\n'.join(self.buffer))

        self.id = None
        self.buffer.clear()

    def add(self, line):

        if line.startswith("<text"):
            self.flush()
            self.id = re.search(r'.*id=\"(\d+[:]\d+)\".*', line).group(1).replace(":", "_")

        self.buffer.append(line)

def process(result_file):

    with DocumentSplitter(result_file) as p:

        for line in fileinput.input():
            p.add(line)

        p.flush()

if __name__ == "__main__":
    process('result_file.zip')