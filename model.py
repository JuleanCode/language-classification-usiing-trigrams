from collections import defaultdict
import os

class LanguageIdentifier:
    def __init__(self):
        self.languages = []
        self.training_data = {}
        self.trigram_models = {}
        self.total_trigrams = {}

    def load_training_data(self, data_path):
        for filename in os.listdir(data_path):
            if filename.endswith(".txt"):
                language = filename.split('.')[0]
                with open(os.path.join(data_path, filename), 'r', encoding='latin-1') as file:
                    text = file.read()
                    self.languages.append(language)
                    self.training_data[language] = text

    def train(self):
        for lang, text in self.training_data.items():
            self.trigram_models[lang] = defaultdict(int)
            self.total_trigrams[lang] = 0

            for i in range(len(text) - 2):
                trigram = text[i:i+3]
                self.trigram_models[lang][trigram] += 1
                self.total_trigrams[lang] += 1

    def language_probability(self, sentence):
        trigrams = [sentence[i:i+3] for i in range(len(sentence) - 2)]
        probabilities = {lang: 0 for lang in self.languages}

        for lang in self.languages:
            prob = 1.0
            for i in range(len(trigrams)):
                trigram = trigrams[i]
                if i >= 2:
                    previous_trigram = trigrams[i-2:i]
                    trigram_prob = (self.trigram_models[lang][trigram] + 1) / (self.total_trigrams[lang] + len(self.trigram_models[lang]))
                    context_prob = (self.trigram_models[lang][previous_trigram[0]] + 1) / (self.total_trigrams[lang] + len(self.trigram_models[lang]))
                    prob *= trigram_prob * context_prob
            probabilities[lang] = prob

        return probabilities

    def identify_language(self, sentence):
        probabilities = self.language_probability(sentence)
        max_lang = max(probabilities, key=probabilities.get)
        sorted_langs = sorted(probabilities, key=probabilities.get, reverse=True)
        
        print("The text is most likely in", max_lang)
        print("Language probabilities:")
        for lang in sorted_langs:
            percentage = (probabilities[lang] / sum(probabilities.values())) * 100
            print(f"{lang}: {percentage:.2f}%")

        return max_lang

    def evaluate(self, test_data):
        total_sentences = len(test_data)
        correct_predictions = 0

        for sentence, actual_language in test_data:
            predicted_language = self.identify_language(sentence)
            if predicted_language == actual_language:
                correct_predictions += 1

        accuracy = correct_predictions / total_sentences
        return accuracy

def main():
    language_identifier = LanguageIdentifier()
    data_path = "data"

    language_identifier.load_training_data(data_path)
    language_identifier.train()

    # Test data is in format [sentence] [language]
    test_data = [
        ("Hoe gaat het vandaag?", "nederlands"),
        ("How are you doing today?", "engels")
    ]

    accuracy = language_identifier.evaluate(test_data)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    input_sentence = input("Enter a sentence: ")
    language_identifier.identify_language(input_sentence)

if __name__ == "__main__":
    main()
