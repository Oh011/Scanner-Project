from flask import Flask, request, send_file, abort
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = "/var/www/files"

@app.route("/download")
def download():
    filename = request.args.get("file")            # taint source
    filename = secure_filename(filename)           # sanitizer: strips ../ and absolute paths
    filepath = os.path.join(BASE_DIR, filename)    # path sink
    real     = os.path.realpath(filepath)          # sanitizer: resolves symlinks
    if not real.startswith(BASE_DIR):              # guard: reject anything outside BASE_DIR
        abort(403)
    return send_file(real)