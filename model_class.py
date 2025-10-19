from keras.models import model_from_json
import numpy as np
import cv2

class CovidModel(object):
    

    
    def __init__(self, model_json_file, model_weights_file):
        self.pred_list=['Covid-19','Normal','Pneumonia']
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
            

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        print("Model loaded from disk")
        self.loaded_model.summary()

    def predict_(self, img):
        self.preds = self.loaded_model.predict(img)[0]
        print(self.preds)
        print(np.argmax(self.preds))
        
        return self.pred_list[np.argmax(self.preds)]

        # return self.preds

# from PIL import Image
# from PIL import ImageFilter
# from palm_detect import extract_palm

if __name__ == '__main__':
    model=CovidModel("model1.json", "model1.h5")
    img=cv2.imread('data/Normal/NORMAL_Train2.png')
    img=cv2.resize(img,(150,150))
    img = cv2.Canny(img,100,200)
    img= np.stack((img,)*3, axis=-1)
    img=np.reshape([img],(1,150,150,3))
    pred=model.predict_(img)
    print(pred)

