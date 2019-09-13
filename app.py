

from flask import Flask, render_template
import os


app = Flask(__name__)


@app.route('/')
def home():

	return render_template("index.html")


@app.route('/about')
def render_about_page():
    return render_template('about.html')

	
	
	



if __name__ == "__main__":
	
	app.run(debug=True,port=os.getenv('PORT',5000))

