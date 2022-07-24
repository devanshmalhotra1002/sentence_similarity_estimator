
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
            clean = [i for i in sentence if i not in string.punctuation]
            clean = "".join(clean)
            return clean

        def cosine_sim(vec1, vec2):
            sim = np.dot(vec1, vec2)/(np.linalg.norm(vec1)*np.linalg.norm(vec2))
            return sim

        def embeddings_similarity(sentences):
            pair = list(itertools.combinations(sentences, 2))
            a = [pair[0] for pair in pair]
            b = [pair[1] for pair in pair]
            pair_df = pd.DataFrame({'a': a, 'b': b})
            pair_df = pair_df.loc[pd.DataFrame(np.sort(
                pair_df[['a', 'b']], 1), index=pair_df.index).drop_duplicates(keep='first').index]
            pair_df = pair_df[pair_df['a'] != pair_df['b']]

            embeddings = spacy.load('en_core_web_sm')

            pair_df['similarity'] = pair_df.apply(
                lambda row: cosine_sim(
                    embeddings(clean(row['a'])).vector,
                    embeddings(clean(row['b'])).vector),
                axis=1
            )
            return pair_df['similarity'][0]
        sentences = [sentence1, sentence2]
        sim = embeddings_similarity(sentences)

        return render_template('approved.html', result=sim)
    else:
        return render_template('value.html')


if __name__ == '__main__':
    app.run(debug = True)

