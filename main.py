import essentia
from essentia.standard import MonoLoader, TensorflowPredictMusiCNN, TensorflowPredict2D
import essentia.standard as es

import numpy as np

audio = MonoLoader(filename="/Users/scottcheung/Code/PersonalCS/SpotifyPlaylistBuilder/Sergei Rachmaninoff, James Levine, Berliner Philharmoniker, Arcadi Volodos - Piano Concerto No. 3 in D Minor, Op. 30- III. Finale. Alla breve.mp3",
                   sampleRate=16000, resampleQuality=4)()

embeddings = TensorflowPredictMusiCNN(
    graphFilename='/Users/scottcheung/Code/PersonalCS/SpotifyPlaylistBuilder/msd-musicnn-1.pb', output="model/dense/BiasAdd")(audio)

model = TensorflowPredict2D(graphFilename='/Users/scottcheung/Code/PersonalCS/SpotifyPlaylistBuilder/danceability-msd-musicnn-1.pb', output="model/Softmax")
predictions = model(embeddings)
mean_probs = np.mean(predictions, axis=0)
print(f"Dance: {mean_probs[0]:.3f}, Not: {mean_probs[1]:.3f}")

# top_n = 5
# averaged_predictions1 = np.mean(predictions1, axis=0)
# msd_labels = ['rock', 'pop', 'alternative', 'indie', 'electronic', 'female vocalists', 'dance', '00s', 'alternative rock', 'jazz', 'beautiful', 'metal', 'chillout', 'male vocalists', 'classic rock', 'soul', 'indie rock', 'Mellow', 'electronica', '80s', 'folk', '90s', 'chill', 'instrumental',
#               'punk', 'oldies', 'blues', 'hard rock', 'ambient', 'acoustic', 'experimental', 'female vocalist', 'guitar', 'Hip-Hop', '70s', 'party', 'country', 'easy listening', 'sexy', 'catchy', 'funk', 'electro', 'heavy metal', 'Progressive rock', '60s', 'rnb', 'indie pop', 'sad', 'House', 'happy']
# for i, l in enumerate(averaged_predictions1.argsort()[-top_n:][::-1], 1):
#     print('{}: {}'.format(i, msd_labels[l]))
