import cv2
import numpy as np

from keras.models import Model, load_model

from keras.layers import Lambda, Input, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3, preprocess_input

WIDTH = 299


class MyModel():

    def __init__(self):
        self.predg_label = None
        self.predg_perc = None
        # labels = pd.read_csv('labels.csv')
        # self.__classes = sorted(list(set(labels['breed'])))
        self.__classes = \
            ['affenpinscher', 'afghan_hound', 'african_hunting_dog', 'airedale', 'american_staffordshire_terrier',
             'appenzeller', 'australian_terrier', 'basenji', 'basset', 'beagle', 'bedlington_terrier',
             'bernese_mountain_dog', 'black-and-tan_coonhound', 'blenheim_spaniel', 'bloodhound', 'bluetick',
             'border_collie', 'border_terrier', 'borzoi', 'boston_bull', 'bouvier_des_flandres', 'boxer',
             'brabancon_griffon', 'briard', 'brittany_spaniel', 'bull_mastiff', 'cairn', 'cardigan',
             'chesapeake_bay_retriever', 'chihuahua', 'chow', 'clumber', 'cocker_spaniel', 'collie',
             'curly-coated_retriever', 'dandie_dinmont', 'dhole', 'dingo', 'doberman', 'english_foxhound',
             'english_setter', 'english_springer', 'entlebucher', 'eskimo_dog', 'flat-coated_retriever',
             'french_bulldog', 'german_shepherd', 'german_short-haired_pointer', 'giant_schnauzer', 'golden_retriever',
             'gordon_setter', 'great_dane', 'great_pyrenees', 'greater_swiss_mountain_dog', 'groenendael',
             'ibizan_hound', 'irish_setter', 'irish_terrier', 'irish_water_spaniel', 'irish_wolfhound',
             'italian_greyhound', 'japanese_spaniel', 'keeshond', 'kelpie', 'kerry_blue_terrier', 'komondor', 'kuvasz',
             'labrador_retriever', 'lakeland_terrier', 'leonberg', 'lhasa', 'malamute', 'malinois', 'maltese_dog',
             'mexican_hairless', 'miniature_pinscher', 'miniature_poodle', 'miniature_schnauzer', 'newfoundland',
             'norfolk_terrier', 'norwegian_elkhound', 'norwich_terrier', 'old_english_sheepdog', 'otterhound',
             'papillon', 'pekinese', 'pembroke', 'pomeranian', 'pug', 'redbone', 'rhodesian_ridgeback', 'rottweiler',
             'saint_bernard', 'saluki', 'samoyed', 'schipperke', 'scotch_terrier', 'scottish_deerhound',
             'sealyham_terrier', 'shetland_sheepdog', 'shih-tzu', 'siberian_husky', 'silky_terrier',
             'soft-coated_wheaten_terrier', 'staffordshire_bullterrier', 'standard_poodle', 'standard_schnauzer',
             'sussex_spaniel', 'tibetan_mastiff', 'tibetan_terrier', 'toy_poodle', 'toy_terrier', 'vizsla',
             'walker_hound', 'weimaraner', 'welsh_springer_spaniel', 'west_highland_white_terrier', 'whippet',
             'wire-haired_fox_terrier', 'yorkshire_terrier']

    def __get_featuares(self, data):
        cnn_model = InceptionV3(include_top=False, input_shape=(WIDTH, WIDTH, 3), weights='imagenet')

        inputs = Input((WIDTH, WIDTH, 3))
        x = inputs
        x = Lambda(preprocess_input, name='preprocessing')(x)
        x = cnn_model(x)
        x = GlobalAveragePooling2D()(x)
        cnn_model = Model(inputs, x)

        features = cnn_model.predict(data, batch_size=64, verbose=1)
        return features

    def __prep_img(self, path):
        X_test = np.zeros((1, WIDTH, WIDTH, 3), dtype=np.uint8)
        X_test[0] = (cv2.resize(cv2.imread(path), (WIDTH, WIDTH)))

        return X_test

    def get_predict(self, path):
        feature = self.__get_featuares(self.__prep_img(path))

        predg = load_model('my_model2.keras').predict(feature, batch_size=128)
        self.predg_label = self.__classes[np.argmax(predg[0])]
        self.predg_perc = round(np.max(predg[0]) * 100)

        print(f"Predicted label: {self.predg_label}")
        print(f"Probability of prediction): {self.predg_perc} %")

# MyModel().get_predict()
