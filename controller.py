import os
import sys
import torch

sys.path.insert(0, os.path.abspath("answer_extraction"))
from extraction_ootb_en import AnswerExtractor

sys.path.insert(0, os.path.abspath("question_generation"))
from question_gen_en import QuestionGenerator

sys.path.insert(0, os.path.abspath("translation"))
from translator_es_en import Translator


class Controller:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.translator = Translator(device)
        print(self.translator.translate("Hola, ¿cómo estás?"))
        self.extractor = AnswerExtractor()
        self.question_generator = QuestionGenerator()



if __name__ == "__main__":
    controller = Controller()
    pass
