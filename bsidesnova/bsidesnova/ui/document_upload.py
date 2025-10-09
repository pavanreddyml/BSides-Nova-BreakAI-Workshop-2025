import ipywidgets as widgets
from IPython.display import display
import os

class DocumentUploadWidget:
    def __init__(self):
        self.uploaded_file_path = None
        self.upload_widget = widgets.FileUpload(
            accept='',  # Accept all file types
            multiple=False,
            description='Upload Document'
        )
        self.output = widgets.Output()
        self.upload_widget.observe(self._on_upload, names='value')
    
    def _on_upload(self, change):
        if change['new']:
            uploaded_file = list(change['new'].values())[0]
            filename = uploaded_file['metadata']['name']
            content = uploaded_file['content']
            
            # Save file to current directory
            self.uploaded_file_path = os.path.join('.', filename)
            
            with open(self.uploaded_file_path, 'wb') as f:
                f.write(content)
            
            with self.output:
                self.output.clear_output()
                print(f"✅ File uploaded successfully: {self.uploaded_file_path}")
    
    def display(self):
        display(self.upload_widget, self.output)
    
    def get_file_path(self):
        return self.uploaded_file_path
