import os
import pandas as pd
import sys
import torch
import nltk

from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util

nltk.download('punkt')

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from answer_extraction.answer_extraction import NER_Extractor
from question_generation.question_gen_en import QuestionGenerator
from translation.translator_es_en import Translator
from distraction_generation.distraction_generator_en import DistractionFinder


ENGLISH_JSON = "data/xquad.en.json"
SPANISH_JSON = "data/xquad.es.json"

def load_qa_data(json_file):
    
    articles = pd.read_json(json_file)

    # turn the json into a list of dictionaries
    articles = [a for a in articles["data"]]

    return articles


def bleu_comparison(original: str, translated: str):
    """
    Compare the generated questions with the dataset questions using BLEU score
    """

    original_tokens = word_tokenize(original.lower())

    translated_tokens = word_tokenize(translated.lower())

    bleu_score = sentence_bleu([original_tokens], translated_tokens)

    bleu_score = bleu_score if bleu_score >= .0001 else 0

    return bleu_score


class Controller:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.translator = Translator(device)
        self.extractor = NER_Extractor()
        self.question_generator = QuestionGenerator()
        self.distraction_finder = DistractionFinder()

    # methods to simplify model calls
    def translate(self, text:str):
        return self.translator.translate(text)

    def extract_answer(self, context: str):     
        self.extractor.change_spacy_initialization(0)
        return self.extractor.process_paragraph(context)
    
    def generate_question(self, answer, context, max_length=64):
        return self.question_generator.generate_question(answer, context)

    def find_distractions(self, context: str, answer: str):
        return self.distraction_finder.example_flow(context, answer)
    
    def gen_qa_pairs(self, context: str, translate = True):
        """
        Generate a question, answer, and distractions for a given context
        :param context: str: the context to generate the question from
        :param translate: bool: whether to translate the context to English
        :return: tuple: the context and a list of tuples containing the question, answer, and distractions
        """
        if translate:
            context = self.translate(context)
        
        answers = self.extract_answer(context)
        
        qa_pairs = []
            
        for a in answers:
            question = self.generate_question(a, context)
            distractions = self.find_distractions(context, a)
            qa_pairs.append((question, a, distractions))
        
        return context, qa_pairs
        


def score_qa_pair(controller, english: dict, spanish: dict, display: bool = False):
    """
    Iterate over the articles and paragraphs in the English and Spanish data to translate,
    the Spansih data is translated to English and then generate questions and answers in English.
    Evaluate 
    """

    # iterate through the articles and paragraphs
    for i, article in enumerate(english[0:1]):
        for j, paragraph in enumerate(article["paragraphs"]):
            if display and j >= 1:
                break

            # get the text to translate
            spanish_context = spanish[i]["paragraphs"][j]["context"]

            translated_context, qa_pairs = controller.gen_qa_pairs(spanish_context)

            # get the correct translation
            target_context = paragraph["context"]

            if display:
                print("--------------------------------------------------\n")
                print(f"Spanish Context: {spanish_context}\n")
                print(f"English Context: {target_context}\n")
                print(f"Translated Context: {translated_context}\n")
                print("--------------------------------------------------\n")

            bleu_score = bleu_comparison(target_context, translated_context)
            print(f"BLEU Score: {bleu_score}\n")

            target_qa = []
            for qas in paragraph["qas"]:
                target_question = qas["question"]
                target_answer = [a["text"] for a in qas["answers"]]
                
                target_qa.append((target_question, target_answer))

            for gen_q, extr_a, extr_dist in qa_pairs:

                if display:
                    print("--------------------------------------------------\n")
                    print(f"Generated-Q: {gen_q}\n")
                    print(f"Extracted-A: {extr_a}\n")
                    print(f"Distractions: {extr_dist}\n")
                    # print(f"Target-QA: {target_qa}\n")
                    print("--------------------------------------------------\n")

            


if __name__ == "__main__":
    controller = Controller()

    english_qa = load_qa_data(ENGLISH_JSON)
    spanish_qa = load_qa_data(SPANISH_JSON)

    # score single pair for display
    score_qa_pair(controller, english_qa, spanish_qa, display=True)
