from itertools import chain

import xmltodict


def get_items(data, key):

    if not isinstance(data, dict):
        raise Exception("not a dict")

    if key not in data:
        return [ ]

    items = data.get(key, None)

    if items is None:
        return [ ]

    if isinstance(items, dict):
        return [ items ]

    if isinstance(items, list):
        return items

    raise AttributeError("key ""{}"" not found in block".format(key))

class AltoXmlParser():

    def __init__(self, line_break=None, block_break=None, page_break=None):
        self.line_break = line_break
        self.block_break = block_break
        self.page_break = page_break

    def document(self, xml_contents):
        return chain.from_iterable(self.pages(xml_content) for xml_content in xml_contents)

    def pages(self, xml_content):
        content = xmltodict.parse(xml_content)
        return chain.from_iterable(
            (self.page(page) for _, page in content['alto']['Layout'].items())
        )

    def page(self, page):

        print_spaces = (x for x in get_items(page, 'PrintSpace'))
        composed_blocks = chain.from_iterable(get_items(ps, 'ComposedBlock') for ps in print_spaces)

        tokens = chain.from_iterable(self.composed_block(x) for x in composed_blocks)

        if self.page_break:
            tokens = chain(tokens, (self.page_break,))

        return tokens

    def composed_block(self, composed_block):
        return chain.from_iterable(
            (self.text_block(x) for x in get_items(composed_block, 'TextBlock'))
        )

    def text_block(self, text_block):
        return chain.from_iterable(
            (self.text_line(x) for x in get_items(text_block, 'TextLine'))
        )

    def text_line(self, text_line):
        tokens = (
            self.token(x) for x in get_items(text_line, 'String')
        )
        if self.line_break is not None:
            tokens = chain(tokens, (self.line_break,))
        return tokens

    def token(self, t):
        return t['@CONTENT']


class AltoXmlParserReversed():

    def __init__(self, xml_content):

        self.content = xmltodict.parse(xml_content)

        assert isinstance(self.content, dict)
        assert 'alto' in self.content

    def pages(self):
        for _, page in self.content['alto']['Layout'].items():
            yield page

    def composed_blocks(self, pages=None):
        for page in (pages or self.pages()):
            for print_space in get_items(page, 'PrintSpace'):
                for composed_block in get_items(print_space, 'ComposedBlock'):
                    yield composed_block

    def text_blocks(self, composed_blocks=None):
        for composed_block in (composed_blocks or self.composed_blocks()):
            for text_block in get_items(composed_block, 'TextBlock'):
                yield text_block

    def text_lines(self, text_blocks=None):
        for text_block in (text_blocks or self.text_blocks()):
            for text_line in get_items(text_block, 'TextLine'):
                yield text_line

    def tokens(self, text_lines=None):
        for text_line in (text_lines or self.text_lines()):
            for token in get_items(text_line, 'String'):
                yield token['@CONTENT']

    def text(self):

        return ' '.join([ x for x in self.tokens() if x != '' ])
