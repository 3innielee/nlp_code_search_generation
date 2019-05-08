from flask import Flask, render_template, request
import json
import requests

app = Flask(__name__, static_url_path='', static_folder='', template_folder='')

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/search', methods=["GET", "POST"])
def search():
	# if request.method == 'POST':
	# 	user_input = request.form # need to check the key(s)
	# 	requests.post() # post to AWS for calculation

	# resp = requests.get() # get the results back from AWS
	# table_content = resp.json()

	return render_template("search.html")


if __name__ == '__main__':
	app.run(port=5000, debug=True)