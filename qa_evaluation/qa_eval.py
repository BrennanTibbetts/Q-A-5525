from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class QA_Evaluator:

    def __init__(self):
        self.embeddings_model = SentenceTransformer('all-MiniLM-l6-v2') 
        model_name = "MaRiOrOsSi/t5-base-finetuned-question-answering"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.QA_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


    def answer_question(self, context: str, question: str):
        """
        Answer a question based on the context provided
        """

        input = f"question: {question} context: {context}"

        encoded_input = self.tokenizer([input],
                                    return_tensors='pt',
                                    max_length=512,
                                    truncation=True)
        
        answer = self.QA_model.generate(input_ids = encoded_input.input_ids,
                                    attention_mask = encoded_input.attention_mask,
                                    max_length=20)
        
        answer = self.tokenizer.decode(answer[0], skip_special_tokens=True)

        return answer


    def calculate_similarity(self, s1, s2):

        #Set up text embedding model:
        embeddings = self.embeddings_model.encode([s1, s2])
        sim_score = cosine_similarity(embeddings[0].reshape(1,-1), embeddings[1].reshape(1,-1))[0][0]
        return sim_score


    def evaluate_qa_pair(self, context, question, target_answer):
        """
        Answer the given question and evaluate the similarity between the generated answer and the target answer
        :param context: str: the context to generate the question from
        :param question: str: the question to answer
        :param target_answer: str: the expected answer
        :return: tuple: the generated answer and the similarity score
        """

        gen_answer = self.answer_question(context, question)

        similarity_score = self.calculate_similarity(gen_answer, target_answer)
        
        return gen_answer, similarity_score
