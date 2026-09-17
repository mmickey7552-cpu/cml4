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

    def convert_img(self,img):
        import numpy as np
        from tensorflow.keras.utils import load_img,img_to_array
        from tensorflow.keras import applications
        from tensorflow.keras.models import load_model,Model
        from tensorflow.image import resize

        #load model as size expected by VGG16
        sample_image=load_img(img,target_size=(224,224))
        sample_img=img_to_array(sample_image)/255.0
        sample=np.expand_dims(sample_img,axis=0)
        #passing through base model to get speicif abstract representation
        self.base_model= applications.VGG16(include_top=False,input_shape=(224,224,3))
        #choosing a layer that outputs (4,4,512)
        layer_name="block5_pool" #last maxpooling layer 
        output_layer=Model(inputs=self.base_model.input,outputs=self.base_model.get_layer(layer_name).output)
        converted_img=output_layer.predict(sample)
        converted_img= converted_img[0]
        resized_sample=resize(converted_img,size=(4,4))
        #get batch dimension representation
        resized=np.expand_dims(resized_sample,axis=0)
        return resized

    def test_sample(self):
        sample1 = self.samples_path+"/sample1.jpg"
        resized=self.convert_img(sample1)
        result=self.restored_model.predict(resized)
        if result[0][0] >= 0.5:
            prediction="dog"
        else:
            prediction="cat"
        self.assertEqual (prediction,"dog","Predicted class is wrong")

if __name__== '__main__':
    unittest.main()


