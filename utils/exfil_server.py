from flask import Flask, request
from cors import CORS

from datetime import datetime

class ExfilServer:
    def __init__(self, host='localhost', port=8080, log_path='exfil.log'):
        self.host = host
        self.port = port
        self.log_path = log_path
        self.app = Flask("ExfilServer")
        CORS(self.app)

    def setup_routes(self):
        @self.app.route('/get-image/<data>', methods=['POST'])
        def exfiltrate(data):
            with open(self.log_path, 'a') as log_file:
                log_file.write(f"Received data: {data}\n")
            return {"status": "success"}, 200

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