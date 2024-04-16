import spacy
from spacy import displacy
import random

class NER_Extractor:
    # both are english. first is fast, second is accurate
    # adapted from https://www.analyticsvidhya.com/blog/2021/06/nlp-application-named-entity-recognition-ner-in-python-with-spacy/

    NER_list = [spacy.load("en_core_web_md"), spacy.load("en_core_web_trf")]
    NER_index = 0

    def change_spacy_initialization(self, index):
        self.NER_index = index
        
    def process_paragraph(self, paragraph_text, display=False):
        tagged_text= self.NER_list[self.NER_index](paragraph_text)
        
        # alternative method if we wanted more direct control of tag types

        good_stuff = [(token.text, token.pos_) for token in tagged_text if token.pos_ in ['NOUN']]
        
        # extract only the token text (ex: 'runs'), not the token type (ex. 'verb')
        good_stuff = [word for word, pos in good_stuff]

        # good_stuff = [token.text for token in tagged_text.ents]
        
        # optional for bugtesting NER
        if(True):
            unique_labels = set()

        # good_stuff = [(token.text, token.pos_) for token in tagged_text]
        # # good_stuff = [(token.text, token.pos_) for token in tagged_text if token.pos_ in ['NOUN', 'VERB', 'PRON']]
        # print('\nThis is the good stuff ')
        # print(good_stuff)

        # optional for bugtesting
        if(display):
            print('\nProcessed Tags')
            for word in tagged_text.ents:
                print('\t'+word.label_, word.text)
                unique_labels.add(word.label_)
                
            print('\nLabel Explaination')
            for label in unique_labels:
                print('\t'+label+' describes '+spacy.explain(str(label)))
        
        # combine NER and raw noun extraction
        final_output = tagged_text.ents + tuple(good_stuff)
        
        # ensure final output is in strings
        final_output = tuple(str(item) for item in final_output)
        
        return random.sample(final_output, 4)
                
    # Work if we wanted to get phrased answers instead of just single subjected words
    # def break_down_sentences(tagged_text):
    #     # based on work conducted within https://subscription.packtpub.com/book/data/9781838987312/2/ch02lvl1sec13/splitting-sentences-into-clauses
    #     for token in tagged_text:
    #         ancestors = [t.text for t in token.ancestors]
    #         children = [t.text for t in token.children]
    #         print(token.text, "\t", token.i, "\t", 
    #             token.pos_, "\t", token.dep_, "\t", 
    #             ancestors, "\t", children)
        
if __name__ == "__main__":
    # Example usage:
    paragraph_text = "The commencement ceremony for OSU graduates will take place on May 20th at Ohio Stadium."
    
    ner_extractor = NER_Extractor()
    ner_extractor.change_spacy_initialization(1)
    ner_extractor.change_spacy_initialization(0)
    print(ner_extractor.process_paragraph(paragraph_text))