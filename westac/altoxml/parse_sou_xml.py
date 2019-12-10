#-*- coding: utf-8 -*-

import sys, os
import re
#from xml.etree.ElementTree import iterparse
import xml.sax

sys.path = [".", "./domain", "./service" ] + sys.path

class TextDocumentHandler(xml.sax.ContentHandler):

    def __init__(self, destination_folder, capture_years):
        self.capture_mode = False
        self.years = capture_years
        self.outfile = None
        self.destination_folder = destination_folder
        return super().__init__()

    def outStartTag(self, name, attrs):
        self.outfile.write("<{0} {1}>".format(name, " ".join(["{0}=\"{1}\"".format(x, attrs.getValue(x)) for x in attrs.getNames()])))

    def outEndTag(self, name):
        self.outfile.write("</{0}>".format(name))
        if name == "text":
            self.outfile.close()
            self.outfile = None
    
    def startElement(self, name, attrs):
        if name == "text":
            print("New SOU: {0}".format(attrs.getValue("id")))
            title = attrs.get('id', None)
            year = int(title[:4])
            if year in self.years:
                self.capture_mode = True
                destination_file = os.path.join(self.destination_folder, title.replace(":", "_") + ".xml")
                self.outfile = open(destination_file, "w", encoding="cp850")
                    
        if self.capture_mode:
            self.outStartTag(name, attrs)


    def characters(self, content):
        if self.capture_mode:
            self.outfile.write(content)
        return super().characters(content)

    def endElement(self, name):
        if self.capture_mode:
            self.outEndTag(name);
        if name == "text":
            self.capture_mode = False


if __name__ == "__main__":


    #source_folder = "M:/#Temp/sou/1970-talet/"
    #destination_folder = "\\\\HORSE.humlab.umu.se\\Delad\\#Temp\\sou\\splits\\"
    source_file = "J:/SOU/sou.xml"
    destination_folder = "J:/SOU/sou_70/"
    years_of_interest = range(1922, 1979)

    parser = xml.sax.make_parser()
    parser.setContentHandler(TextDocumentHandler(destination_folder, years_of_interest))
    parser.parse(open(source_file,"r", encoding="cp850"))

    #with open(destination_file, "w") as outfile:
        
    #    for (event, node) in iterparse(source_file, ['start', 'end']):
            
    #        if event == 'start':
    #            if node.tag != 'text':
    #                continue
    #            print (node.attrib)
            


