import spacy
import numpy as np

# Make sure to run: python -m spacy download en_core_web_md to load the model

class DistractionFinder:
    def __init__(self, model='en_core_web_md'):
        self.nlp = spacy.load(model)

    def find_distraction_word(self, passage, target_words, target_similarity):
        """
        Find the noun or verb in the passage whose average similarity to the target words
        is closest to the specified target similarity score.

        Parameters:
        passage (str): The text passage
        target_words (list of str): The target words (assumed to be nouns or verbs)
        target_similarity (float): The target similarity score to approximate

        Returns:
        str: The noun or verb with the closest average similarity score to the target similarity
        """
        
        doc = self.nlp(passage)
        doc_text = [token.text for token in doc]
        for word in target_words:
            if word not in doc_text:
                return f"Target word '{word}' not in passage."

        target_tokens = [token for token in doc if token.text in target_words and token.pos_ in ['NOUN', 'VERB', 'PROPN']]

        if not target_tokens:
            return "None of the target words are nouns or verbs."

        distraction_word = None
        distraction_difference = float('inf')

        for token in doc:
            if token.pos_ not in ['NOUN', 'VERB', 'PROPN'] or token.text in target_words:
                continue

            similarities = [target_token.similarity(token) for target_token in target_tokens]
            average_similarity = np.mean(similarities)

            difference = abs(average_similarity - target_similarity)

            if difference < distraction_difference:
                distraction_word = token.text
                distraction_difference = difference

        return [distraction_word, distraction_difference]

    def example_flow(self, passage, answer):
        target_words = [answer]
        target_similarity = 0.95

        first_distraction_word, _ = self.find_distraction_word(passage, target_words, target_similarity)
        target_similarity = 0.4
        target_words = [target_words[0], first_distraction_word]  # This ensures the original word is included
        second_distraction_word, _ = self.find_distraction_word(passage, target_words, target_similarity)

        target_similarity = 0.95
        target_words = [first_distraction_word, second_distraction_word]  # Now using only the generated words
        third_distraction_word, _ = self.find_distraction_word(passage, target_words, target_similarity)

        return first_distraction_word, second_distraction_word, third_distraction_word


passage = """KERRVILLE, TX—Exasperated with the view from the place they were standing 
to observe the astronomical event, local spectators complained Monday that really tall guy 
Matt Everett was blocking everyone's view of the total solar eclipse. 
“Goddammit, this thing only lasts a few minutes—can't he at least sit down?” 
said Garett Pointer, 5' 8", who was seen craning his neck around Everett, 6' 5", 
in an attempt to get a better look as the moon passed between the earth and the sun. 
“This is my last chance to see once of these things until 2044, 
and I wind up stuck behind Abe fucking Lincoln. Just my luck. 
And the asshole isn't even paying attention! He's been staring at his phone the whole time.” 
At press time, the total eclipse had reportedly become even more difficult to view after 
Everett's girlfriend decided to perch atop his shoulders."""

finder = DistractionFinder()
answer = "Garett"
first, second, third = finder.example_flow(passage, answer)
print(f"Answer: {answer}")
print(f"First distraction word: {first}")
print(f"Second distraction word: {second}")
print(f"Third distraction word: {third}")