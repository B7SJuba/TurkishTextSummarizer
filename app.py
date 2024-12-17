from flask import Flask
from flask import Flask, render_template, request
#from main import text_summarize #importing the function from main
from main import textRank
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("Homepage.html")

@app.route("/", methods=["GET","POST"])
def index():
    #For textRank function
    if request.method == "POST":
        #Retrieve the text from the text area
        user_input = request.form.get("inputText")
        slider_value = int(request.form.get("number"))
        processed_result = textRank(user_input, slider_value)

        return render_template("Homepage.html", inputText=user_input, output_text=processed_result[0],
                               inLength=processed_result[1], outLength=processed_result[2], html_fig1=processed_result[3],
                               html_fig2 = processed_result[4], html_fig3 = processed_result[5])
    else:
        return render_template("Homepage.html")


if __name__ == "__main__":
    app.run(debug=True, port=8000) #default port is 5000 we can choose any port we want


'''
    #For the method using word frequencies....
    if request.method == "POST":
        #Retrieve the text from the text area
        user_input = request.form.get("inputText")
        slider_value = int(request.form.get("number"))
        processed_result = text_summarize(user_input, slider_value)

        return render_template("Homepage.html", inputText=user_input, output_text=processed_result[0],
                               inLength=processed_result[1], outLength=processed_result[2])
    else:
        return render_template("Homepage.html")


'''