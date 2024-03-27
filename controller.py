import os
import sys
from torch import device

sys.path.insert(0, os.path.abspath("answer-extraction"))
from extraction_ootb_en import AnswerExtractor

sys.path.insert(0, os.path.abspath("question_generation"))
from question_gen_en import QuestionGenerator

sys.path.insert(0, os.path.abspath("translation"))
# from translation import Translator


class Controller:
    def __init__(self, device):
        self.extractor = AnswerExtractor(device)
        self.question_generator = QuestionGenerator()
        # self.translator = Translator


if __name__ == "__main__":
    pass
