import spacy
import numpy as np
import random

class Answer_Extractor:

    #constructs array from raw_text: should not be called
    def generate_answers(raw_text, num_questions):
        sample_answer_arr = [[],[],[]]
        answer_arr = [[],[],[]]

        #populate array from text
        #tokens can be modified for different parts of speech
        for token in raw_text:
            if token.pos_ == 'NOUN' and token not in sample_answer_arr[0]:
                sample_answer_arr[0].append(token)
            if token.pos_ == 'PROPN'and token not in sample_answer_arr[1]:
                sample_answer_arr[1].append(token)
            if token.pos_ == 'VERB' and token not in sample_answer_arr[2]:
                sample_answer_arr[2].append(token)

        if (len(sample_answer_arr[0]) < num_questions or len(sample_answer_arr[1])  < num_questions or len(sample_answer_arr[2])  < num_questions):
            print ("Not enough potential answers from passage: number of questions too high")
            return

        #generate num_question amount of possible answers for each part of speech
        filled = False
        words_filled = 0
        while(not filled):
            #populate array with unique answers
            noun_word = sample_answer_arr[0][random.randint(0, len(sample_answer_arr[0]) - 1)]
            pnoun_word = sample_answer_arr[1][random.randint(0, len(sample_answer_arr[1]) - 1)]
            verb_word = sample_answer_arr[2][random.randint(0, len(sample_answer_arr[2]) - 1)]
            if (noun_word not in answer_arr[0] and pnoun_word not in answer_arr[1] and verb_word not in answer_arr[2]):
                answer_arr[0].append(noun_word)
                answer_arr[1].append(pnoun_word)
                answer_arr[2].append(verb_word)
                words_filled += 1
            
            if (words_filled == num_questions):
                filled = True

        return answer_arr

    #returns array holding words in form [[nouns], [proper nouns], [verbs]] each containing num_questions entries 
    def process_paragraph(text, num_questions):
        #load attribute list
        nlp = spacy.load('en_core_web_sm')

        #generate answer list
        prompt_text = nlp(text)
        answer_arr = Answer_Extractor.generate_answers(prompt_text, num_questions)
        return answer_arr

#sample text for testing
arr = Answer_Extractor.process_paragraph("""KERRVILLE, TX—Exasperated with the view from the place they were standing 
to observe the astronomical event, local spectators complained Monday that really tall guy 
Matt Everett was blocking everyone's view of the total solar eclipse. 
“Goddammit, this thing only lasts a few minutes—can't he at least sit down?” 
said Garett Pointer, 5' 8", who was seen craning his neck around Everett, 6' 5", 
in an attempt to get a better look as the moon passed between the earth and the sun. 
“This is my last chance to see once of these things until 2044, 
and I wind up stuck behind Abe fucking Lincoln. Just my luck. 
And the asshole isn't even paying attention! He's been staring at his phone the whole time.” 
At press time, the total eclipse had reportedly become even more difficult to view after 
Everett's girlfriend decided to perch atop his shoulders.""", 4)
print(arr)




