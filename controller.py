import os
import sys
import torch

sys.path.insert(0, os.path.abspath("answer_extraction"))
from extraction_ootb_en import AnswerExtractor

sys.path.insert(0, os.path.abspath("question_generation"))
from question_gen_en import QuestionGenerator

sys.path.insert(0, os.path.abspath("translation"))
from translation_es_en import Translator


class Controller:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.extractor = AnswerExtractor(device)
        self.question_generator = QuestionGenerator(device)
        self.translator = Translator(device)


if __name__ == "__main__":
    pass
