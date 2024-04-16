import os
import pandas as pd
import sys
import torch
import nltk

from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util

nltk.download('punkt')

sys.path.insert(0, os.path.abspath("answer_extraction"))
from answer_extraction import NER_Extractor

sys.path.insert(0, os.path.abspath("question_generation"))
from question_gen_en import QuestionGenerator

sys.path.insert(0, os.path.abspath("translation"))
from translator_es_en import Translator

sys.path.insert(0, os.path.abspath("distraction_generation"))
from distraction_generator_en import DistractionFinder


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


def semantic_comparison(generated_questions: list[str], dataset_questions: list[str]):
    """
    Compare the generated questions with the dataset questions using the model based on semantic similarity   
    """

    model = SentenceTransformer('all-MiniLM-L6-v2')

    generated_embeddings = model.encode(generated_questions)
    dataset_embeddings = model.encode(dataset_questions)

    gen_embedding = sum(generated_embeddings) / len(generated_embeddings)
    dataset_embedding = sum(dataset_embeddings) / len(dataset_embeddings)

    semantic_similarities = util.pytorch_cos_sim(gen_embedding, dataset_embedding)

    return semantic_similarities.item()


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

    def find_distractions(self, passage: str, answer: str):
        return self.distraction_finder.example_flow(passage, answer)


def score_qa_pair(controller, english: dict, spanish: dict, display: bool = False):
    """
    Iterate over the articles and paragraphs in the English and Spanish data to translate,
    the Spansih data is translated to English and then generate questions and answers in English.
    Evaluate 
    """

    # iterate through the articles and paragraphs
    for i, article in enumerate(english):
        for j, paragraph in enumerate(article["paragraphs"]):
            if display and j >= 1:
                break

            # get the text to translate
            spanish_context = spanish[i]["paragraphs"][j]["context"]
            translated_context = controller.translate(spanish_context)

            # get the correct translation
            target_context = paragraph["context"]

            if display:
                print("--------------------------------------------------\n")
                print(f"Spanish Context: {spanish_context}\n")
                print(f"English Context: {target_context}\n")
                print(f"Translated Context: {translated_context}\n")
                print("--------------------------------------------------\n")

            blue_score = bleu_comparison(target_context, translated_context)
            print(f"BLEU Score: {blue_score}\n")

            target_qa = []
            for qas in paragraph["qas"]:
                target_question = qas["question"]
                target_answer = [a["text"] for a in qas["answers"]]
                
                target_qa.append((target_question, target_answer))

            extracted_answers = controller.extract_answer(translated_context)


            for extracted_answer in extracted_answers:
                
                generated_question = controller.generate_question(extracted_answer, translated_context)
                
                extracted_distractions = controller.find_distractions(translated_context, extracted_answer)

                # bleu_score = bleu_comparison(target_context, translated_context)

                gen_qa = []
                dataset_qa = []

                if display:
                    print("--------------------------------------------------\n")
                    print(f"Generated-Q: {generated_question}\n")
                    print(f"Extracted-A: {extracted_answer}\n")
                    print(f"Distractions: {extracted_distractions}\n")
                    print(f"Target-QA: {target_qa}\n")
                    print("--------------------------------------------------\n")
                        
                # similarity = semantic_comparison(gen_qa, dataset_qa)

                # if display:
                #     print("--------------------------------------------------\n")
                #     print(f"QA Pair Semantic Similarity: {similarity}\n")

            
            # bleu_score = bleu_comparison(target_context, translated_context)


if __name__ == "__main__":
    controller = Controller()

    english_qa = load_qa_data(ENGLISH_JSON)
    spanish_qa = load_qa_data(SPANISH_JSON)

    # score single pair for display
    score_qa_pair(controller, english_qa, spanish_qa, display=True)
