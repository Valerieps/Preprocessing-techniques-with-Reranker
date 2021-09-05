# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def tf_idf(space_left, input):

    # remove stop words
    stop_words = set(stopwords.words('english'))
    text_tokens = word_tokenize(input)
    tokens_without_sw = [word for word in text_tokens if not word in stop_words]
    input_no_sw = " ".join(tokens_without_sw)

    # tf-idf
    vectorizer = TfidfVectorizer(strip_accents='ascii', analyzer='word')
    matrix = vectorizer.fit_transform([input_no_sw])
    dense = matrix.todense().tolist()
    features = vectorizer.get_feature_names()
    df = pd.DataFrame(dense, columns=features)
    scores = df.iloc[0]
    ranking = list(zip(scores, features))
    ranking.sort(reverse=True)

    # cropar no tamanho do space left
    space_left = int(space_left - space_left * 0.25)

    # Make string out of ranking
    new_body = ""
    for ranking, word in ranking[:space_left]:
        new_body += word + " "

    return new_body


input = "Science Mathematics Physics The hot glowing surfaces of stars emit energy in the form of electromagnetic radiation It is a good approximation to assume that the emissivity e is equal to 1 for these surfaces. Find the radius of the star Rigel, the bright blue star in the constellation Orion that radiates energy at a rate of 2.7 x 10^32 W and has a surface temperature of 11,000 K. Assume that the star is spherical. Use Ïƒ =... show more Follow 3 answers Answers Relevance Rating Newest Oldest Best Answer: Stefan-Boltzmann law states that the energy flux by radiation is proportional to the forth power of the temperature: q = Îµ Â· Ïƒ Â· T^4 The total energy flux at a spherical surface of Radius R is Q = qÂ·Ï€Â·RÂ² = ÎµÂ·ÏƒÂ·T^4Â·Ï€Â·RÂ² Hence the radius is R = âˆš ( Q / (ÎµÂ·ÏƒÂ·T^4Â·Ï€) ) = âˆš ( 2.7x10+32 W / (1 Â· 5.67x10-8W/mÂ²K^4 Â· (1100K)^4 Â· Ï€) ) = 3.22x10+13 m Source (s):http://en.wikipedia.org/wiki/Stefan_bolt...schmiso Â· 1 decade ago0 18 Comment Schmiso, you forgot a 4 in your answer. Your link even says it: L = 4pi (R^2)sigma (T^4). Using L, luminosity, as the energy in this problem, you can find the radius R by doing sqrt (L/ (4pisigma (T^4)). Hope this helps everyone. Caroline Â· 4 years ago4 1 Comment (Stefan-Boltzmann law) L = 4pi*R^2*sigma*T^4 Solving for R we get: => R = (1/ (2T^2)) * sqrt (L/ (pi*sigma)) Plugging in your values you should get: => R = (1/ (2 (11,000K)^2)) *sqrt ( (2.7*10^32W)/ (pi * (5.67*10^-8 W/m^2K^4))) R = 1.609 * 10^11 m? Â· 3 years ago0 1 Comment Maybe you would like to learn more about one of these? Want to build a free website? Interested in dating sites? Need a Home Security Safe? How to order contacts online?"
ranking = tf_idf(200, input)
print(len(ranking))
print(ranking[:10])