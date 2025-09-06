from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route("/")
def home():
    return "<h1>Welcome to Vision-Guard</h1><p>Here you will open PDFs securely with face unlock.</p>"

if __name__ == "__main__":
    app.run(debug=True)
