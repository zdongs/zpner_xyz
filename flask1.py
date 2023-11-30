from flask import Flask, request, jsonify, render_template
from model_utils.one_tuili import pre_model_tokenizer,call_entities

app = Flask(__name__)

model, tokenizer = pre_model_tokenizer()

@app.route("/")
def index():
    return render_template("test.html")


@app.route("/predict", methods=["POST"])
def predict_from():
    if request.method == "POST":
        data = request.form.get("input_data")  # 接受POST请求中的JSON
        entities = call_entities(data, model, tokenizer)
        return render_template("test.html", input_data=data, result=entities)
    else:
        return f"你来到了没有知识的荒原……"


@app.route("/api/hello", methods=["GET"])
def hello():
    return jsonify({"message": "Hello, World!"})


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
