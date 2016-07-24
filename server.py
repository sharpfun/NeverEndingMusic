from flask import Flask, request, render_template, url_for, make_response
app = Flask(__name__)
from gen import sample_musicxml


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template("form_submit.html")
    if request.method == 'POST':
        mxml = sample_musicxml(request.form['source'])
        response = make_response(mxml)
        response.headers["Content-Disposition"] = "attachment; filename=out.xml"
        return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12345, debug=True)
