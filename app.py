from flask import Flask, request, jsonify,render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import joblib
import pandas as pd
from model.my_model import MyModel
model = joblib.load('model/gradient_boosting_model.pkl')
app = Flask(__name__)
CORS(app)  # 允许跨域请求
# 上传文件保存的目录
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
@app.route('/data', methods=['POST'])
def handle_data():
    try:
        # if 'files' in request.files:
            # 获取上传的文件数据
        print('loading')
        file = request.files['files']
        # 处理文件数据，例如保存到本地或者进行其他操作
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # 读取上传的文件内容并转换为模型输入格式（假设为CSV文件）
        df = pd.read_csv(file_path, encoding='gbk')
        df['max月统筹金占总比例'] = df['月统筹金额_MAX'] /df['统筹支付金额_SUM']
        df = df.fillna(0)
        # 提取特征列
        X = df[['max月统筹金占总比例', '月统筹金额_MAX', '月就诊次数_MAX', '本次审批金额_SUM', '月药品金额_AVG']]

        # 进行预测
        predictions = model.predict(X)

        # 将预测结果添加到文件中
        df['res'] = predictions

        # 保存带有预测结果的文件
        df.to_csv(file_path, index=False)

        return jsonify({'message': 'File uploaded successfully'})
        # else:
        #     return '未找到上传的文件', 400
    except Exception as e:
        print('处理请求时发生异常：', e)
        return jsonify({'error': '处理请求时发生异常'}),600

if __name__ == '__main__':
    app.run(debug=True, port=8081)
