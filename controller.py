import os
import pandas as pd
import sys
import torch

from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
from sentence_transformers import SentenceTransformer, util

sys.path.insert(0, os.path.abspath("answer_extraction"))
from extraction_ootb_en import AnswerExtractor

sys.path.insert(0, os.path.abspath("question_generation"))
from question_gen_en import QuestionGenerator

sys.path.insert(0, os.path.abspath("translation"))
from translator_es_en import Translator


ENGLISH_JSON = "data/xquad.en.json"
SPANISH_JSON = "data/xquad.es.json"

def load_qa_data(json_file):
    
    articles = pd.read_json(json_file)

    # turn the json into a list of dictionaries
    articles = [a for a in articles["data"]]

    return articles[11:12]


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
        #print(self.translator.translate("Hola, ¿cómo estás?"))
        self.extractor = AnswerExtractor()
        self.question_generator = QuestionGenerator()

    # methods to simplify model calls
    def translate(self, text:str):
        return self.translator.translate(text)
    
    def generate_questions(self, text: str, num_questions: int = 4):
        return self.question_generator.generate_questions(text, 
                                                          num_questions=num_questions, 
                                                          temperature=3.0)

    def extract_answer(self, question: str, context: str):     
        return self.extractor.extract_answer(question, context)


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

            target_qa = []
            for qas in paragraph["qas"]:
                target_question = qas["question"]
                target_answer = [a["text"] for a in qas["answers"]]
                
                target_qa.append((target_question, target_answer))

            # generate question
            generated_questions = controller.generate_questions(translated_context, num_questions=len(target_qa))

            bleu_score = bleu_comparison(target_context, translated_context)

            if display:
                print("--------------------------------------------------\n")
                print(f"Spanish Context: {spanish_context}\n")

                # use BLEU score to evaluate the translation
                print(f"Translation BLEU: {bleu_score}\n")
                
                print(f"English Context: {target_context}\n")
                print(f"Translated Context: {translated_context}\n")
                print("--------------------------------------------------\n")

            gen_qa = []
            # extract answer
            for question in generated_questions:
                answer = controller.extract_answer(question, translated_context)

                gen_qa.append(f"Q: {question}, A:{answer}")
                if display:
                    print(f"Generated-Q: {question}")
                    print(f"Generated-A: {answer}\n")

            if display:
                print("--------------------------------------------------\n")

            dataset_qa = []
            for qa in target_qa:
                
                dataset_qa.append(f"Q: {qa[0]}, A:{qa[1][0]}")
                
                if display:
                    print(f"Target-Q: {qa[0]}")
                    print(f"Target-A: {qa[1][0]}\n")
                    

            similarity = semantic_comparison(gen_qa, dataset_qa)

            if display:
                print("--------------------------------------------------\n")
                print(f"QA Pair Semantic Similarity: {similarity}\n")


if __name__ == "__main__":
    controller = Controller()

    english_qa = load_qa_data(ENGLISH_JSON)
    spanish_qa = load_qa_data(SPANISH_JSON)

    # score single pair for display
    score_qa_pair(controller, english_qa, spanish_qa, display=True)
