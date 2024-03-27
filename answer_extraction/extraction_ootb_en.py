from transformers import (
    AutoTokenizer,
    AutoModelWithLMHead,
    AutoModelForQuestionAnswering,
    pipeline,
)
import torch


class AnswerExtractorEnglish:

    def __init__(self, device):

        model_name = "t5-small"

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelWithLMHead.from_pretrained(model_name).to(self.device)

    def extract_answer(self, question, context):

        input_text = f"question: {question} context: {context}"
        features = self.tokenizer(input_text, return_tensors="pt")

        output = self.model.generate(
            input_ids=features["input_ids"].to(self.device),
            attention_mask=features["attention_mask"].to(self.device),
            max_length=100,
        )

        output = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return output


class AnswerExtractor:

    def __init__(self, device):

        model_name = "deepset/xlm-roberta-large-squad2"

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(
            self.device
        )

    def extract_answer(self, question, context):

        pipe = pipeline(
            "question-answering",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
        )

        output = pipe(question=question, context=context)

        return output["answer"]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    question = "What is the capital of France?"
    context = "France is a country in Europe. Its capital is Paris."

    extractor = AnswerExtractorEnglish(device)
    print("English: ", extractor.extract_answer(question, context))
    question_es = "¿Cuál es la capital de Francia?"
    context_es = "Francia es un país de Europa. Su capital es París."

    extractor = AnswerExtractor(device)
    print("Question (es): ", question_es)
    print("Context: ", context_es)
    print(
        "Multilingual: ",
        extractor.extract_answer(question_es, context_es),
    )
    print("Question (en): ", question)
    print("Context: ", context)
    print(
        "Multilingual: ",
        extractor.extract_answer(question, context),
    )
