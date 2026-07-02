from flask import Flask, request, send_file

app = Flask(__name__)

@app.route("/download")
def download():
    filename = request.args.get("file")        # taint source
    filepath = "/var/www/files/" + filename    # string concat — no sanitization
    return send_file(filepath)  