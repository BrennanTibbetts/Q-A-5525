class Controller:
    def __init__(self,
                 translator=None,
                 extractor=None,
                 question_generator=None,
                 distraction_finder=None):
        """
        Initialize the controller with the necessary components
        :param translator: Translator: the translator component
        :param extractor: NER_Extractor: the answer extraction component
        :param question_generator: QuestionGenerator: the question generation component
        :param distraction_finder: DistractionFinder: the distraction generation component
        """

        self.translator = translator
        self.extractor = extractor 
        self.question_generator = question_generator
        self.distraction_finder = distraction_finder

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
