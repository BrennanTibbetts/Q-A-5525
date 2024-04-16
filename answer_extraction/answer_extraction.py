import spacy
from spacy import displacy
import random

class NER_Extractor:
    # both are english. first is fast, second is accurate
    # adapted from https://www.analyticsvidhya.com/blog/2021/06/nlp-application-named-entity-recognition-ner-in-python-with-spacy/

    def __init__(self):
        self.NER_list = ["en_core_web_md", "en_core_web_trf"]
        self.NER_index = 0
        self.nlp = spacy.load(self.NER_list[self.NER_index])

    def change_spacy_initialization(self, index):
        self.NER_index = index
        
    def process_paragraph(self, paragraph_text, display=False):
        # Process the text to extract entities
        tagged_text = self.nlp(paragraph_text)

        # Extract proper nouns for the alternative method
        good_stuff = [token.text for token in tagged_text if token.pos_ == 'PROPN']
        
        # Extract entities
        entities = [str(entity) for entity in tagged_text.ents]

        # Check if the entities list is shorter than 4
        if len(entities) < 4:
            needed = 4 - len(entities)
            additional_samples = random.sample(good_stuff, min(needed, len(good_stuff)))
            final_output = entities + additional_samples
        else:
            final_output = random.sample(entities, 4)

        # Optional debugging and display of NER labels
        if display:
            unique_labels = set((entity.label_ for entity in tagged_text.ents))
            print('\nProcessed Tags')
            for entity in tagged_text.ents:
                print('\t' + entity.label_, entity.text)
            print('\nLabel Explanation')
            for label in unique_labels:
                print('\t' + label + ' describes ' + spacy.explain(str(label)))

        return final_output
                
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