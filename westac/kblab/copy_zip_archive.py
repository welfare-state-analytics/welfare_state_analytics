import os
import zipfile

import click

# pylint: disable=no-value-for-parameter, chained-comparison

@click.command()
@click.argument('source', )
@click.argument('target', )
@click.argument('startyear', type=click.IntRange(1700, 2030))
@click.argument('endyear', type=click.IntRange(1700, 2030))
def copyto(source, target, startyear, endyear):

    with zipfile.ZipFile(source, "r") as source_archive:

        filenames = source_archive.namelist()

        with zipfile.ZipFile(target, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as target_archive:

            for filename in filenames:

                parts = os.path.split(filename)[0].split('-')

                if len(parts) < 2:
                    continue

                year = int(parts[1])

                if year >= startyear and year <= endyear:

                    content = source_archive.read(filename)

                    target_archive.writestr(filename, data=content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)

                    print("added: {}".format(filename))

if __name__ == "__main__":
    copyto()
