
from music_genre_classifier.inference import predict_genre_from_file


if __name__ == "__main__":
    genre, prob, conf_text = predict_genre_from_file("pop.wav")
    print(f"{conf_text}{genre} (p={prob:.2f})")