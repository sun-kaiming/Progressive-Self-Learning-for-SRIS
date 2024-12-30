from flask import Flask, request

app = Flask(__name__)


@app.route('/', methods=['POST'])
def receive_text():
    input_text = request.form['input-box']
    # Do something with input_text
    return 'Input received: {}'.format(input_text)


if __name__ == '__main__':
    app.run()
