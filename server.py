from flask import Flask, request, render_template, url_for
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template("form_submit.html")
    if request.method == 'POST':
        return request.form['source']

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12345, debug=True)
