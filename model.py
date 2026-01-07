"""
Logistic回归预测模型核心逻辑
"""
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression


class LogisticPredictionModel:
    """Logistic回归预测模型类"""
    
    def __init__(self):
        """初始化模型"""
        self.model = LogisticRegression()
        # 使用真实训练得到的系数
        # 变量顺序：猝倒、打鼾、性格变化、幻觉、遗尿、性别、嗜睡家族史、睡瘫、年龄
        self.coefficients = np.array([
            3.717851,   # 猝倒 (有)
            -1.986443,  # 打鼾 (有)
            1.343246,   # 性格变化 (有)
            0.888900,   # 幻觉 (有)
            -0.790445,  # 遗尿 (有)
            -0.766936,  # 性别 (女)
            0.513365,   # 嗜睡家族史 (有)
            0.509313,   # 睡瘫 (有)
            -0.051412   # 年龄, 岁
        ])
        self.intercept = -0.930143  # 截距
        
    def fit(self, X, y):
        """训练模型"""
        self.model.fit(X, y)
        self.coefficients = self.model.coef_[0]
        self.intercept = self.model.intercept_[0]
        
    def predict_logit(self, X):
        """计算Logit值"""
        if self.coefficients is None or self.intercept is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        # Logit = β0 + β1*X1 + β2*X2 + ... + βn*Xn
        logit = self.intercept + np.dot(X, self.coefficients)
        return logit
    
    def predict_proba(self, X):
        """计算概率值p"""
        if self.coefficients is None or self.intercept is None:
            raise ValueError("模型尚未训练，请先调用fit方法")
        
        logit = self.predict_logit(X)
        # p = 1 / (1 + exp(-logit))
        p = 1 / (1 + np.exp(-logit))
        return p
    
    def predict_diagnosis(self, X, threshold=0.905):
        """根据阈值预测诊断结果"""
        p = self.predict_proba(X)
        diagnosis = "诊断" if p >= threshold else "不诊断"
        return diagnosis, p
    
    def calculate_from_inputs(self, inputs, threshold=0.905):
        """
        从输入数据计算Logit和p值，并返回诊断结果
        
        参数:
            inputs: 输入特征数组，顺序为：
                    [猝倒(有=1), 打鼾(有=1), 性格变化(有=1), 幻觉(有=1), 
                     遗尿(有=1), 性别(女=1), 嗜睡家族史(有=1), 睡瘫(有=1), 年龄(岁)]
            threshold: 诊断阈值，默认0.905
            
        返回:
            dict: 包含logit, p值, 诊断结果的字典
        """
        inputs_array = np.array(inputs).reshape(1, -1)
        
        # 确保输入特征数量正确
        if len(inputs) != len(self.coefficients):
            raise ValueError(f"输入特征数量应为{len(self.coefficients)}，但得到{len(inputs)}")
        
        logit = self.predict_logit(inputs_array)[0]
        p = self.predict_proba(inputs_array)[0]
        diagnosis, _ = self.predict_diagnosis(inputs_array, threshold)
        
        return {
            'logit': logit,
            'p_value': p,
            'diagnosis': diagnosis,
            'threshold': threshold
        }
