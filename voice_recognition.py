import pandas as pd
import librosa

from data_extraction import *
from data_processing import *
from model_extraction import *


class Recognizer:
    def __init__(self, model='best_model'):
        self.__model = load_model(name=model)
        df = read_raw_files()
        # print(df.label.unique())
        _, _, _, _, self.__lb = split_df(df, transform=True)
        self.__opt = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
        self.__model.compile(loss='categorical_crossentropy', optimizer=self.__opt, metrics=['accuracy'])

    def __helper(self, audiofile, start_from=0.5):
        duration=2.5
        X, sample_rate = librosa.load(audiofile, res_type='kaiser_fast', duration=duration, sr=22050 * 2,
                                      offset=start_from)
        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
        featurelive = mfccs
        livedf2 = featurelive
        livedf2 = pd.DataFrame(data=livedf2)
        livedf2 = livedf2.stack().to_frame().T
        twodim = np.expand_dims(livedf2, axis=2)
        livepreds = self.__model.predict(twodim,
                                         batch_size=32,
                                         verbose=1)
        livepreds1 = livepreds.argmax(axis=1)
        liveabc = livepreds1.astype(int).flatten()
        predictions = (self.__lb.inverse_transform((liveabc)))
        return predictions

    def recognize(self, audiofiles, start_from=0.5):
        preds = {}
        for audiofile in audiofiles:
            preds[audiofile] = self.__helper(audiofile, start_from)[0]
        return preds

    def get_model(self):
        return self.__model.summary()