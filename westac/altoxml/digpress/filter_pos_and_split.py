#-*- coding: utf-8 -*-

import os
import glob
import re
import nltk
import codecs

# We use xml SAX for XML file parsing
import xml.sax

# Class that writes words to a text file, and rolls over to new file when specified chunk is reached
class SouWriter():

    def __init__(self, destination, title, chunk_size):
        self.outfile = None                 # current output file
        self.title = title                  # title of current XML text tag (SOU document)
        self.file_counter = 0               # current part of file counter
        self.word_counter = 0               # counter within current part-of-file (param)
        self.chunk_size = chunk_size        # max number of words of each chunk (param)
        self.destination = destination      # destination folder (param)
        return super().__init__()

    # create a new filename based on current title and current value part-of-file counter
    def create_filename(self): 
        return os.path.join(self.destination, "{0}_{1:04d}.txt".format(self.title.replace(":", "_"), self.file_counter))

    # write word to disk and increment word-counter. Rollover to next file if word count equals chunk size
    def output(self, token):
        if self.outfile == None or self.word_counter % self.chunk_size == 0:
            self.rollover()
        self.word_counter += 1
        self.outfile.write(token + " ")

    # close file if open
    def close(self):
        if self.outfile != None:
            self.outfile.close()

    # Rolls over to next file within current title (SOU file)
    def rollover(self):
        self.close()
        self.file_counter += 1
        self.word_counter = 0
        self.outfile = open(self.create_filename(), "w", encoding="utf-8")

# Class that filters words in XML based on part-of-speach (POS) and writes output to text files of a given max size
class SouXmlHandler(xml.sax.ContentHandler):

    # Option is a dictionary that must be populated as:
    #   source                          Source file of folder where SOU XML file(s) reside
    #   destination                     Destination folder where splitted text files are places
    #   years_of_interest               List of years to process i.e. years not listed are ingnored 
    #   pos_of_interest                 List such as [ "NN" ] with POS of interest
    #   use_lemma                       Flag indicating that 
    #   chunk_size                      Number of words per resulting text file
    #   lemma_or_content                Flag that 1) use lemma as output otherwise) original word

    def __init__(self, options):
        self.options = options
        self.writer = None
        self.capture = False
        self.lemma = None
        return super().__init__()

    # Specified STEM written to resulting textfile if use_lemma is tru
    # Rule #1: Select last lemma if more than one (seperated by vertical bar)
    # Rule #2: Assume that both PREFIX and SUFFIX matches "^\|([\w]+\.{2}(nn|vb|...)\|)+$"
    def get_lemma(self, attrs):

        if not self.options["use_lemma"]:
            return None
        lemma = self.create_lexeme_stem(attrs.getValue("lemma"))
        if lemma == None:
            lemma = self.create_compound_lexeme_stem(attrs.getValue("prefix"), attrs.getValue("suffix"))
        return lemma

    # Creates a compound word out a the stems of the two sub-parts
    def create_compound_lexeme_stem(self, prefix, suffix):

        try:
            if prefix != "|" and suffix != "|":
                #x = [ x.split("..")[0] for x in prefix.split('|')[1:-1]][-1]
                #y = [ x.split("..")[0] for x in suffix.split('|')[1:-1]][-1]
                #return x + y
                return ''.join([ y.split('|')[1:-1][-1].split("..")[0] for y in [prefix, suffix]])
        except:
            pass

        return None

    # Select word (if exists) in lemma tag
    # Rule #1: Select FIRST lemma if more than one
    def create_lexeme_stem(self, lemma):

        try:
            return lemma.split("|")[1] if lemma != "" and lemma != "|" else None
        except:
            return None

    # Action that takes place at start of each tag
    # Only "text" and "w" is handled. A "text" tag indicates a new SOU-document, and "w" a word.
    # Only "text" tags with a year found in self.options["years_of_interest"] is processed
    # Only "w" tags with a part-of-speach found in self.options["years_of_interest"] is processed
    # A "w" is ignored of the year is missing in the self.options["pos_of_interest"] is captured
    def startElement(self, name, attrs):

        if name == "corpus" or name == "text":

            if 'id' in attrs:
                title = attrs.getValue('id')
                year = int(title[:4])
            else:
                title = os.path.splitext(os.path.basename(self.options["source_file"]))[0]
                year = 0

            if year == 0 or year in self.options["years_of_interest"]:
                self.writer = SouWriter(self.options["destination"], title, self.options["chunk_size"])

            print("{0} corpus: {1}".format("Found" if self.writer != None else "Skipped", title))

        if name == "w":

            self.capture = False
            self.lemma = None

            if self.writer != None:

                self.capture = len(self.options["pos_of_interest"]) == 0 or \
                    attrs.getValue("pos") in self.options["pos_of_interest"]

                if self.capture:

                    self.lemma  = self.get_lemma(attrs)


    # Action that takes place at end of each tag
    def endElement(self, name):

        # Close current file at end of "text"
        if name == "corpus" or name == "text":

            if self.writer != None:
                self.writer.close();
                self.writer = None

    # Text action. Write lemma or word to file. 
    def characters(self, content):
        if self.capture:
            #if self.lemma != None and self.lemma != content:
            #    print("{0}:{1}".format(content, self.lemma))
            self.writer.output(self.lemma.lower() if self.lemma != None else content.lower())
            self.capture = False
        #return super().characters(content)

# Error handler (not used for now)
class SouErrorHandler(xml.sax.handler.ErrorHandler):

    def error(self, exception):
        pass

    def fatalError(self, exception):
        pass

    def warning(self, exception):
        pass

class SouProcessService():

    # Process each specified source file in sequence
    def process(self, files, options):

        for source_file in files:

            print("Processing file {0}...".format(source_file))
            options.update({ 'source_file': source_file })
            service = SouXmlHandler(options)
            parser = xml.sax.make_parser()
            parser.setContentHandler(service)
            parser.parse(open(source_file,"r", encoding="utf-8"))

if __name__ == "__main__":

    # Note: The following data must be configures to each specific case
    # Place select different location for source and destination
    options = {
        "source": "C:/temp/pope/",             # test
        "destination": "C:/temp/pope/splits",  # test
        "years_of_interest": range(1922, 1979),     # Ignore "text" tags having an ID not found in list
        "pos_of_interest": [ "NN", "NNS", "NNP", "NNPS" ],          # See http://spraakbanken.gu.se/korp/markup/msdtags.html
        "use_lemma": True,                          # True if we want to use lemma instead of content
        "chunk_size": 1000                          # Number of words to write in each file
    }
        
    # Verify that destination folder exists
    if not os.path.isdir(options["destination"]):
        print("Please create destination folder first")
        exit()

    if os.path.isdir(options["source"]):
        # source is a folder
        files = [ f for f in glob.glob(os.path.join(options["source"],"*.xml")) ]
    else:
        # source is a file
        files = [ options["source"] ]

    # Start the process
    SouProcessService().process(files, options)


    
