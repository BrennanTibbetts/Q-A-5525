from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


class QuestionGenerator:
    def __init__(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "voidful/context-only-question-generator"
        )
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            "voidful/context-only-question-generator"
        ).to(device)

    def generate_questions(
        self, context: str, num_questions: int = 4, temperature: float = 1.0
    ):
        """
        Generate questions from a given context using the model
        """

        input_ids = self.tokenizer(context, return_tensors="pt").input_ids
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=50,
            num_return_sequences=num_questions,
            num_beams=num_questions,
            num_beam_groups=num_questions,  # number of groups to divide the questions into
            temperature=temperature,    # higher temoerature means more randomness
            do_sample=False,
            length_penalty=-1,  # penalize longer questions because it gets incoherent
            diversity_penalty=5.0,  # encourages diversity in the questions
            no_repeat_ngram_size=2  # avoid repeating bigrams to ensure diversity
        )
        questions = list(
            {
                self.tokenizer.decode(output, skip_special_tokens=True)
                for output in outputs
            }
        )
        return questions
