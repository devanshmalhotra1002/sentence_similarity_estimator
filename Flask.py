
from flask import Flask, request, render_template, redirect
import pandas as pd
import numpy as np
import string
import itertools
import spacy


app = Flask(__name__)


@app.route('/')
def land():
    return render_template('land.html')


@app.route('/Similarity', methods=['POST', 'GET'])
def sentence_similarity():
    if request.method == 'POST':
        sentence1 = str(request.form['sentence1'])
        sentence2 = str(request.form['sentence2'])

        def clean(sentence):
            clean = "".join([i for i in sentence if i not in string.punctuation])
            return clean

        def cosine_sim(vec1, vec2):
            sim = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
            return sim

        def embeddings_similarity(sentences):
            combination = list(itertools.combinations(sentences, 2))
            a = [pair[0] for pair in combination]
            b = [pair[1] for pair in combination]
            df = pd.DataFrame({'Sentence 1': a, 'Sentence 2': b})

            embeddings = spacy.load('en_core_web_sm')

            df['similarity'] = df.apply(
                lambda row: cosine_sim(
                    embeddings(clean(row['Sentence 1'])).vector,
                    embeddings(clean(row['Sentence 2'])).vector),
                axis=1
            )
            return df['similarity'][0]
        sentences = [sentence1, sentence2]
        sim = embeddings_similarity(sentences)

        return render_template('approved.html', result=sim)
    else:
        return render_template('value.html')


if __name__ == '__main__':
    app.run(debug = True)

