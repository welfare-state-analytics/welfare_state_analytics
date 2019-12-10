#-*- coding: utf-8 -*-

import sys, os
import re
import xml.sax
from xml.sax.saxutils import escape
from xml.sax.saxutils import quoteattr

sys.path = ["." ] + sys.path

# TODO Change encoding to UTF-8?
# Class that handles the XML parsing
class TextDocumentHandler(xml.sax.ContentHandler):

    def __init__(self, destination_folder, capture_years):
        self.capture_mode = False
        self.years = capture_years
        self.outfile = None
        self.destination_folder = destination_folder
        return super().__init__()

    def outStartTag(self, name, attrs):
        # Write tag attributes, escape and quote tag value to handle non-valid XML characters
        self.outfile.write("<{0} {1}>".format(name, " ".join(["{0}={1}".format(x, quoteattr(attrs.getValue(x))) for x in attrs.getNames()])))

    def outEndTag(self, name):
        # Write end tag, close file if end-of-text is reached...
        self.outfile.write("</{0}>".format(name))
        if name == "text":
            self.outfile.close()
            self.outfile = None
    
    # Handler for start of XML tag
    def startElement(self, name, attrs):
        
        # The "text" tag indicates a new SOU-document
        if name == "text":
            print("New SOU: {0}".format(attrs.getValue("id")))
            # Extract document ID and year
            title = attrs.get('id', None)
            year = int(title[:4])
            # Only process selected years...
            if year in self.years:
                # Set capture flag...
                self.capture_mode = True
                # Open new file for the new SOU-document...
                destination_file = os.path.join(self.destination_folder, title.replace(":", "_") + ".xml")
                self.outfile = open(destination_file, "w", encoding="cp850")

        # Write tag if we are capturing an SOU-document
        if self.capture_mode:
            self.outStartTag(name, attrs)

    # Handler for content (the text inside a tag)
    def characters(self, content):
        if self.capture_mode:
            # Only write if we have found an SOU-document we want to capture.
            # Escape content to handle non-valid XML characters
            self.outfile.write(escape(content))
        return super().characters(content)

    def endElement(self, name):
        if self.capture_mode:
            self.outEndTag(name);
        if name == "text":
            # End of SOU-document, stop capturing data...
            self.capture_mode = False


if __name__ == "__main__":

    # Please set values to match your specific setup..
    source_file = "J:/SOU/sou.xml"
    destination_folder = "J:/SOU/xml_splits/"
    years_of_interest = range(1979, 1997)

    # Create XML parser, set handler and open file for processing...
    parser = xml.sax.make_parser()
    parser.setContentHandler(TextDocumentHandler(destination_folder, years_of_interest))
    parser.parse(open(source_file,"r", encoding="cp850"))



