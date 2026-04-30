from flask import Flask

'''It creates an instance of the Flask class,
 which will be your WSGI application'''

##WSGI APPLICATION
app = Flask(__name__)

@app.route("/")
def welcome():
    return "Welcome to this Flask course. This should be an amazing course. This is a fantastic course"

@app.route("/index")
def index():
    return "Welcome to the index page"

#This is the Empty point of the particular application
if __name__=="__main__":
    app.run(debug=True)