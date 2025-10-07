from adversarial_lab.arsenal.adversarial.whitebox import *
from adversarial_lab.arsenal.adversarial.blackbox import *

import os
import tensorflow as tf
import numpy as np
from PIL import Image

class Selector:
    def __init__(self, root_path):
        self.root_path = root_path

    def attacker_selector(self, config):
        category = config["category"]
        dataset = config["dataset"]
        attack = config["attack"]
        attack_params = config["attack_params"]
        image_class = config["image_class"]
        image_name = config["image_name"]
        target_class_idx = config["target_class_idx"]        

        if category == "whitebox":
            model, input_shape, preprocess_fn, decode_preds = self.get_model(dataset)
            img_arr = self.get_image("imagenet", image_class, image_name, input_shape)

            def preprocess(sample, *args, **kwargs):
                input_sample = tf.cast(sample, dtype=tf.float32)
                if len(input_sample.shape) == 2:
                    input_sample = tf.expand_dims(input_sample, axis=-1)
                    input_sample = tf.image.grayscale_to_rgb(input_sample)

                elif len(input_sample.shape) == 3 and input_sample.shape[-1] == 1:
                    input_sample = tf.image.grayscale_to_rgb(input_sample)

                input_tensor = tf.convert_to_tensor(input_sample, dtype=tf.float32)
                resized_image = tf.image.resize(input_tensor, input_shape[:2])
                batch_image = tf.expand_dims(resized_image, axis=0)
                return preprocess_fn(batch_image)
            
            def predict_fn(samples, *args, **kwargs):
                samples = np.array(samples)
                preds = model.predict(samples, verbose=0)
                return [pred for pred in preds]
            
            if attack == "Fast Sign Gradient Method":
                attacker = FastSignGradientMethodAttack(model, 
                                                        preprocess=preprocess, 
                                                        **attack_params)
            elif attack == "Projected Gradient Descent":
                attacker = ProjectedGradientDescentAttack(model, 
                                                          preprocess=preprocess, 
                                                          **attack_params)
            elif attack == "Carlini Wagner":
                attacker = CarliniWagnerAttack(model, 
                                                 preprocess=preprocess, 
                                                 **attack_params)
            elif attack == "Deep Fool":
                attacker = DeepFoolAttack(model, 
                                          preprocess=preprocess, 
                                          **attack_params)
            elif attack == "Smooth Fool":
                attacker = SmoothFoolAttack(model, 
                                            preprocess=preprocess, 
                                            **attack_params)
                

            return {
                "attacker": attacker,
                "sample": img_arr,
                "input_shape": input_shape,
                "preprocess_fn": preprocess,
                "predict_fn": predict_fn,
                "decode_preds": decode_preds,
                "target_class": target_class_idx,
            }
            
        if category == "blackbox":
            pred_fn, preprocess_fn, decode_preds, input_shape = self.get_model(dataset)
            img_arr = self.get_image(dataset, image_class, image_name, input_shape)

            if attack == "Finite Difference":
                attacker = FiniteDifferenceAttack(pred_fn, 
                                        **attack_params)
                
            elif attack == "NES":
                attacker = NESAttack(pred_fn, 
                                    **attack_params)
                
            elif attack == "RGF":
                attacker = RGFAttack(pred_fn, 
                                    **attack_params)
                
            elif attack == "SPSA":
                attacker = SPSAAttack(pred_fn, 
                                     **attack_params)

            return {
                "attacker": attacker,
                "sample": img_arr,
                "input_shape": input_shape,
                "preprocess_fn": preprocess_fn,
                "predict_fn": pred_fn,
                "decode_preds": decode_preds,
                "target_class": target_class_idx,
            }

    def get_model(self, dataset):
        if dataset == "inception_v3":
            from tensorflow.keras.applications import InceptionV3
            from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
            model = InceptionV3(weights='imagenet')
            input_shape = (299, 299, 3)
            return model, preprocess_input, decode_predictions, input_shape
        elif dataset == "resnet":
            from tensorflow.keras.applications import ResNet50
            from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
            model = ResNet50(weights='imagenet')
            input_shape = (224, 224, 3)
            return model, preprocess_input, decode_predictions, input_shape
        elif dataset == "mobilenet":
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
            model = MobileNetV2(weights='imagenet')
            input_shape = (224, 224, 3)
            return model, preprocess_input, decode_predictions, input_shape
        elif dataset == "mnist_digits":
            model = tf.keras.models.load_model(os.path.join(self.root_path, "assets/models/mnist_digits.h5"))

            def pred_fn(samples, *args, **kwargs):
                samples = np.array(samples)
                preds = model.predict(samples, verbose=0)
                return [pred for pred in preds]
            
            def decode_predictions(preds, top=1):
                results = []
                for pred in preds:
                    top_indices = np.argsort(pred)[-top:][::-1]
                    top_scores = [(str(i), float(pred[i])) for i in top_indices]
                    results.append(top_scores)
                return results
            
            input_shape = (28, 28, 1)
            return pred_fn, preprocess_input, decode_predictions, input_shape

    def get_image(self, directory, image_class, image_name, input_shape):
        path = os.path.join(self.root_path, "assets/images", directory, image_class, image_name)
        img = Image.open(path).convert("RGB")
        img = img.resize((input_shape[1], input_shape[0]))
        img_arr = np.array(img)
        return img_arr
