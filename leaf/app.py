import os
import io
import torch
import torchvision.transforms as transforms
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch.nn as nn
from torchvision.models import swin_t

# 配置中文疾病名称映射
disease_names = {
    "Tea algal leaf spot": "茶藻斑病",
    "Brown Blight": "茶褐斑病",
    "Gray Blight": "茶灰斑病",
    "Helopeltis": "茶黑刺蝽",
    "Red spider": "茶红蜘蛛",
    "Green mirid bug": "茶绿盲蝽",
    "Healthy leaf": "健康茶叶"
}

# 设备配置
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {DEVICE}")

# 图像预处理转换
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 创建 FastAPI 应用
app = FastAPI(title="茶叶疾病分类器")

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 创建模型类
class TeaLeafModel(nn.Module):
    def __init__(self, num_classes=7):
        super(TeaLeafModel, self).__init__()
        self.model = swin_t(weights=None)
        self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        
    def forward(self, x):
        return self.model(x)

# 加载模型
def load_model():
    model = TeaLeafModel(num_classes=7)
    model_path = os.path.join(os.getcwd(),'leaf', 'best_model.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在")
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# 预测函数
def predict_image(model, image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = nn.functional.softmax(outputs, dim=1)
            _, predicted_idx = torch.max(outputs, 1)
        
        # 获取预测结果
        idx_to_class = {
            0: "Tea algal leaf spot",
            1: "Brown Blight",
            2: "Gray Blight",
            3: "Helopeltis",
            4: "Red spider", 
            5: "Green mirid bug",
            6: "Healthy leaf"
        }
        
        predicted_class = idx_to_class[predicted_idx.item()]
        chinese_class = disease_names[predicted_class]
        probability = probabilities[0][predicted_idx].item()
        
        # 获取所有类别的概率
        all_probs = {}
        for i, prob in enumerate(probabilities[0]):
            class_name = idx_to_class[i]
            chinese_name = disease_names[class_name]
            all_probs[chinese_name] = float(prob)
        
        # 按概率从高到低排序
        sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "prediction": chinese_class,
            "probability": float(probability),
            "all_probabilities": dict(sorted_probs)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测过程中出错: {str(e)}")

# 全局模型变量
model = None

@app.on_event("startup")
async def startup_event():
    global model
    print("正在加载模型...")
    try:
        model = load_model()
        print("模型加载成功!")
    except Exception as e:
        print(f"模型加载失败: {str(e)}")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    global model
    if model is None:
        raise HTTPException(status_code=500, detail="模型未加载")
    
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="只接受图片文件")
    
    content = await file.read()
    result = predict_image(model, content)
    return JSONResponse(content=result)

# 挂载静态文件
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
