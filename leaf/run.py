import os
import uvicorn
import webbrowser
from threading import Timer

def open_browser():
    webbrowser.open("http://localhost:8000")

if __name__ == "__main__":
    # 确保static目录存在
    os.makedirs("static", exist_ok=True)
    
    # 检查模型文件是否存在
    model_path = os.path.join(os.getcwd(), 'best_model.pth')
    if not os.path.exists(model_path):
        print(f"警告: 模型文件 {model_path} 不存在!")
    
    # 开启浏览器
    Timer(1.5, open_browser).start()
    
    # 启动服务器
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
