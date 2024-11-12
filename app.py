from flask import Flask, render_template, request, redirect, url_for, session, flash,jsonify
import os
from myword2vec import Function

app = Flask(__name__)
app.secret_key = '1123'  # 用於 session 管理

# 儲存上傳模型的位置
UPLOAD_FOLDER = 'uploaded_models'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# 全域變數用於儲存已載入的模型
left_model = None
right_model = None
function = Function()

# 上傳模型路由


@app.route("/upload_model", methods=["POST"])
def upload_model():
    if 'model_file' not in request.files:
        flash("沒有檔案被上傳", "error")
        return redirect(url_for('result'))

    file = request.files['model_file']
    if file.filename == '':
        flash("檔案名稱不可為空", "error")
        return redirect(url_for('result'))

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)
    flash(f"檔案 {file.filename} 已成功上傳", "success")
    return redirect(url_for('result'))
# 載入左或右側模型路由


@app.route("/load_model/<side>", methods=["POST"])
def load_model(side):
    model_name = request.form.get('selected_model')

    if not model_name:
        flash("未選擇模型", "error")
        return redirect(url_for('result'))

    model_path = os.path.join(UPLOAD_FOLDER, model_name)
    try:
        if side == "left":
            function.LoadNormalModel(model_path)
            session['selected_left_model'] = model_name  # 記錄左側選擇的模型
            flash(f"{model_name} 模型已成功載入到左側", "success")
            # 提取高頻詞
           
        elif side == "right":
            function.LoadBestModel(model_path)
            session['selected_right_model'] = model_name  # 記錄右側選擇的模型
            flash(f"{model_name} 模型已成功載入到右側", "success")
             # 提取高頻詞
           

        session['all_words'] = function.GetNormalModelWords()
        
    except Exception as e:
        flash(f"載入模型失敗: {e}", "error")
    return redirect(url_for('result'))


def get_file_list():
    return [f for f in os.listdir('.') if os.path.isfile(f)]



@app.route("/", methods=["GET", "POST"])
def index():
    files = get_file_list()

    if request.method == "POST":
        selected_file = request.form.get("file_select")
        if selected_file:
            function.load_data(selected_file)
            return redirect(url_for("result"))

    return render_template("index.html", files=files)


@app.route("/generate_wordcloud/<side>", methods=["POST"])
def generate_wordcloud(side):
    # 獲取選擇的關鍵字
    keyword = request.form.get("keyword")
    if not keyword:
        flash("請選擇一個關鍵字來生成詞雲", "error")
        return redirect(url_for("result"))


    # 將選擇的關鍵字存入 session
    if side == "left":
        session['selected_left_keyword'] = keyword
    elif side == "right":
        session['selected_right_keyword'] = keyword

    # 根據選擇的模型（左或右）找到關鍵字的相似詞
    try:
        if side == "left" :
            function.GetNormalWordCloud(keyword, 30)
            function.GetNormalModel2D()
        elif side == "right" :
            function.GetBestWordCloud(keyword, 30)
            function.GetBestModel2D()
          
    except Exception as e:
        flash(f"生成詞雲時發生錯誤: {e}", "error")

    return redirect(url_for("result"))

@app.route("/calculate_word", methods=["POST"])
def calculate_word():
    word1 = request.form.get("word1")
    word2 = request.form.get("word2")
    word3 = request.form.get("word3")

    # 更新 session 中的詞語選擇
    session['word1'] = word1
    session['word2'] = word2
    session['word3'] = word3

    

    try:
        result_Normal_word = function.calNormalVec(word1,word2,word3)
        result_Best_word = function.calBestVec(word1,word2,word3)

        flash(f"詞雲: {result_Normal_word} -  {result_Best_word}")
        session['result_Normal_word'] = result_Normal_word 
        session['result_Best_word'] = result_Best_word 
    except KeyError:
        session['result_word'] = "無法計算，請選擇其他詞語。"
    return redirect(url_for('result'))


@app.route("/update_slider_position", methods=["POST"])
def update_slider_position():
    data = request.get_json()
    position = int(data["position"])
    
    # 在這裡處理滑桿位置，例如進行計算
    session['slider_position'] = position  # 可以選擇將位置存入 session

    left_high_freq_words = [word for word, _ in function.Get_Countwords(position)]
    right_high_freq_words = [word for word, _ in function.Get_Countwords(position)]
    session['left_high_freq_words'] = left_high_freq_words
    session['right_high_freq_words'] = right_high_freq_words
    

    # 返回 JSON 響應
    return jsonify({
        "left_high_freq_words": left_high_freq_words,
        "right_high_freq_words": right_high_freq_words,
        "status": "success"
    })


@app.route("/result")
def result():
    models = [f for f in os.listdir(UPLOAD_FOLDER) if f.endswith('.model')]
    selected_left_model = session.get('selected_left_model', '')
    selected_right_model = session.get('selected_right_model', '')
    left_high_freq_words = session.get('left_high_freq_words', [])
    right_high_freq_words = session.get('right_high_freq_words', [])
    selected_left_keyword = session.get('selected_left_keyword', '')
    selected_right_keyword = session.get('selected_right_keyword', '')
    all_words = session.get('all_words', [])
    word1 = session.get('word1', '')
    word2 = session.get('word2', '')
    word3 = session.get('word3', '')
    result_Normal_word = session.get('result_Normal_word', '') 
    result_Best_word = session.get('result_Best_word', '') 
    slider_position = session.get('slider_position', 50)  # 默認位置為中間


    return render_template(
        "result.html",
        left_wordcloud_url="static/images/Normal_wordcloud.png",
        right_wordcloud_url="static/images/Best_wordcloud.png",
        left_2D_url="static/images/Normal_2D.png",
        right_2D_url="static/images/Best_2D.png",
        models=models,
        selected_left_model=selected_left_model,
        selected_right_model=selected_right_model,
        left_high_freq_words=left_high_freq_words,
        right_high_freq_words=right_high_freq_words,
        selected_left_keyword=selected_left_keyword,
        selected_right_keyword=selected_right_keyword,
        all_words=all_words,  # 傳遞詞語列表
        word1=word1,
        word2=word2,
        word3=word3,
        result_Normal_word =result_Normal_word ,
        result_Best_word =result_Best_word,
        slider_position=slider_position
    )


if __name__ == "__main__":
    app.run(debug=True)
