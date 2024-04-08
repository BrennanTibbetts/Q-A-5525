# Adapted from
# https://www.analyticsvidhya.com/blog/2021/06/nlp-application-named-entity-recognition-ner-in-python-with-spacy/

import spacy
from spacy import displacy

class NER_Extractor:
    # both are english. first is fast, second is accurate
    NER_list = [spacy.load("en_core_web_sm"), spacy.load("en_core_web_trf")]
    NER_index = 0

    def change_spacy_initialization(self, index):
        self.NER_index = index
        
    def process_paragraph(self, paragraph_text):
        # paragraph_text = "The Indian Space Research Organisation or is the national space agency of India, headquartered in Bengaluru. It operates under Department of Space which is directly overseen by the Prime Minister of India while Chairman of ISRO acts as executive of DOS as well."
        tagged_text= self.NER_list[self.NER_index](paragraph_text)
        unique_labels = set()

        print('\nProcessed Tags')
        for word in tagged_text.ents:
            print('\t'+word.label_, word.text)
            unique_labels.add(word.label_)
            
        print('\nLabel Explaination')
        for label in unique_labels:
            print('\t'+label+' describes '+spacy.explain(str(label)))
            
if __name__ == "__main__":
    # Example usage:
    paragraph_text = "The Indian Space Research Organisation or is the national space agency of India, headquartered in Bengaluru. It operates under Department of Space which is directly overseen by the Prime Minister of India while Chairman of ISRO acts as executive of DOS as well."
    
    ner_extractor = NER_Extractor()
    ner_extractor.change_spacy_initialization(0)
    ner_extractor.process_paragraph(paragraph_text)