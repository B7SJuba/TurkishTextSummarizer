#Views/Routes are kind of the end of a url

from flask import Flask,Blueprint, render_template, request

views = Blueprint(__name__,"views")

@views.route("/")
def home():
    return render_template("Homepage.html")

@views.route("/", methods=['GET', 'POST'])
def index():
    if request.method=='POST':
        #Retrieve the text from the text area
        input_text = request.form.get("inputText")
        print(input_text) #for verification
        return input_text