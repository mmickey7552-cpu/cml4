import unittest

class TestModel(unittest.TestCase):
    restored_model=None
    base_model=None
    top_model_complete= 'model.keras'
    samples_path="samples"
    #we are allowed to specail functions in test class , setUp and tearDown . 
    #setUp to set the loadings etc which dont need to be repeated for all tests and tear down to bring down that setup. here teardown is not used. convert_img is just helper function to reprocess the image in expected dim .
    def setUp(self):
        from tensorflow.keras.models import load_model
        self.restored_model=load_model(self.top_model_complete)

    def convert_img(self, img):
        import numpy as np
        from tensorflow.keras.utils import load_img, img_to_array
        from tensorflow.keras import applications
    
        sample_image = load_img(img, target_size=(150, 150))
        sample_img = img_to_array(sample_image) / 255.0          # (150, 150, 3)
        sample = np.expand_dims(sample_img, axis=0)              # (1, 150, 150, 3)
        self.base_model = applications.VGG16(include_top=False, weights='imagenet',
                                             input_shape=(150, 150, 3))
        return self.base_model.predict(sample)                   # (1, 4, 4, 512)

    def test_sample(self):
        sample1 = self.samples_path+"/sample1.jpg"
        resized=self.convert_img(sample1)
        result=self.restored_model.predict(resized)
        print(result[0][0])
        if result[0][0] >= 0.5:
            prediction="dog"
        else:
            prediction="cat"
        self.assertEqual (prediction,"dog","Predicted class is wrong")

if __name__== '__main__':
    unittest.main()


