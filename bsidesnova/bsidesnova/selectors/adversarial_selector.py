import os

import numpy as np
from PIL import Image


class AdversarialSelector:
    def __init__(self, root_path):
        self.root_path = root_path
        
    def attacker_selector(self, config):
        import tensorflow as tf
        from adversarial_lab.arsenal.adversarial.whitebox import FastSignGradientMethodAttack, ProjectedGradientDescentAttack, CarliniWagnerAttack, DeepFoolAttack, SmoothFoolAttack
        from adversarial_lab.arsenal.adversarial.blackbox import FiniteDifferenceAttack, NESAttack, RGFAttack, SPSAAttack
        category = config["category"]
        model_name = config["model"]

        attack = config["attack"]
        attack_params = config["attack_params"]

        image_array = config.get("image_array", None) 
        image_class = config["image_class"]
        image_name = config["image_name"]
        target_class_idx = config["target_class_idx"]
        


        # Get all model related data
        model_data = self.get_model(model_name)
        model = model_data["model"]
        preprocess_fn = model_data["preprocess_input"]
        decode_preds = model_data["decode_predictions"]
        input_shape = model_data["input_shape"]



        # Get image Array
        dataset_dir_map = {
            "inception": "imagenet",
            "resnet": "imagenet",
            "mobilenet": "imagenet",
            "mnist_digits": "digits"
        }
        if image_array is not None:
            img_arr = image_array
            img_arr = tf.image.resize(img_arr, input_shape[:2]).numpy()
        else:
            img_arr = self.get_image(directory=dataset_dir_map[model_name], 
                                     image_class=image_class, 
                                     image_name=image_name, 
                                     input_shape=input_shape)


        # Get Attacker
        if category == "whitebox":
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
            
            if attack == "Fast Sign Gradient Method":
                attacker = FastSignGradientMethodAttack
            elif attack == "Projected Gradient Descent":
                attacker = ProjectedGradientDescentAttack
            elif attack == "Carlini Wagner":
                attacker = CarliniWagnerAttack
            elif attack == "Deep Fool":
                attacker = DeepFoolAttack
            elif attack == "Smooth Fool":
                attacker = SmoothFoolAttack
                

            return {
                "attacker": attacker,
                "model": model,
                "preprocess": preprocess,
                "attack_params": attack_params,
                "input_shape": input_shape,
                "sample": img_arr,
                "decode_preds": decode_preds,
                "target_class": target_class_idx,
            }
            
        if category == "blackbox":
            if attack == "Finite Difference":
                attacker = FiniteDifferenceAttack
            elif attack == "NES":
                attacker = NESAttack
            elif attack == "RGF":
                attacker = RGFAttack
            elif attack == "SPSA":
                attacker = SPSAAttack

            return {
                "attacker": attacker,
                "model": model,
                "preprocess": preprocess,
                "attack_params": attack_params,
                "input_shape": input_shape,
                "sample": img_arr.astype(int),
                "decode_preds": decode_preds,
                "target_class": target_class_idx,
            }

    def get_model(self, dataset):
        
        if dataset == "inception":
            from tensorflow.keras.applications import InceptionV3
            from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
            model = InceptionV3(weights='imagenet')
            input_shape = (299, 299, 3)
            return {
                "model": model,
                "preprocess_input": preprocess_input,
                "decode_predictions": decode_predictions,
                "input_shape": input_shape
            }
        elif dataset == "resnet":
            from tensorflow.keras.applications import ResNet50
            from tensorflow.keras.applications.resnet import preprocess_input, decode_predictions
            model = ResNet50(weights='imagenet')
            input_shape = (224, 224, 3)
            return {
                "model": model,
                "preprocess_input": preprocess_input,
                "decode_predictions": decode_predictions,
                "input_shape": input_shape
            }
        elif dataset == "mobilenet":
            from tensorflow.keras.applications import MobileNetV2
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
            model = MobileNetV2(weights='imagenet')
            input_shape = (224, 224, 3)
            return {
                "model": model,
                "preprocess_input": preprocess_input,
                "decode_predictions": decode_predictions,
                "input_shape": input_shape
            }
        elif dataset == "mnist_digits":
            from tensorflow import tf
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
            return {
                "model": pred_fn,
                "preprocess_input": preprocess_input,
                "decode_predictions": decode_predictions,
                "input_shape": input_shape
            }

    def get_image(self, directory, image_class, image_name, input_shape):
        path = os.path.join(self.root_path, "assets/images", directory, image_class, image_name)
        img = Image.open(path).convert("RGB")
        img = img.resize((input_shape[1], input_shape[0]))
        img_arr = np.array(img)
        return img_arr
