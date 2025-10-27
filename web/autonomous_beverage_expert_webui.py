"""
自主现制饮品研发专家系统Web UI（基于真实品牌营养补充剂）
真正的AI研发专家，从营养补充剂选择到再加工全流程自主实现
集成市场分析和商业价值评估功能
支持多种营养补充剂智能融合和协同效应分析
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

# 导入品牌营养补充剂数据集
from brand_nutritional_supplement_dataset import (
    BRAND_NUTRITIONAL_SUPPLEMENTS,
    BASE_BEVERAGES,
    FLAVOR_ADJUSTERS,
    TARGET_CONSUMERS
)

# 导入专家系统
from autonomous_beverage_formulation_expert import AutonomousBeverageExpert

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 创建专家系统实例
expert_system = AutonomousBeverageExpert()

# Flask应用
app = Flask(__name__)
CORS(app)

# HTML模板
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>自主现制饮品研发专家系统</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        header {
            text-align: center;
            margin-bottom: 30px;
            color: white;
        }
        
        h1 {
            font-size: 2.5rem;
            margin-bottom: 10px;
        }
        
        .subtitle {
            font-size: 1.2rem;
            opacity: 0.9;
        }
        
        .card {
            background: white;
            border-radius: 10px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            padding: 25px;
            margin-bottom: 25px;
            transition: transform 0.3s ease;
        }
        
        .card:hover {
            transform: translateY(-5px);
        }
        
        h2 {
            color: #667eea;
            margin-bottom: 20px;
            font-size: 1.5rem;
        }
        
        .form-group {
            margin-bottom: 20px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
        }
        
        select, input, textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e8ed;
            border-radius: 8px;
            font-size: 1rem;
            transition: border-color 0.3s ease;
        }
        
        select:focus, input:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 25px;
            border-radius: 8px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease;
        }
        
        button:hover {
            transform: scale(1.05);
        }
        
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .result {
            background: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            border-left: 4px solid #667eea;
        }
        
        .loading {
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .nutrient-list {
            list-style: none;
        }
        
        .nutrient-item {
            padding: 8px;
            border-bottom: 1px solid #eee;
        }
        
        .nutrient-item:last-child {
            border-bottom: none;
        }
        
        .nutrient-name {
            font-weight: 600;
            color: #333;
        }
        
        .scores {
            display: flex;
            justify-content: space-between;
            margin: 15px 0;
        }
        
        .score-item {
            text-align: center;
            padding: 10px;
            background: #e3f2fd;
            border-radius: 8px;
            flex: 1;
            margin: 0 5px;
        }
        
        .score-value {
            font-size: 1.2rem;
            font-weight: bold;
            color: #667eea;
        }
        
        .preparation-steps {
            background: #e3f2fd;
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            white-space: pre-wrap;
        }
        
        .market-analysis {
            background: linear-gradient(135deg, #ff9a9e 0%, #fad0c4 100%);
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
        }
        
        .synergy-analysis {
            background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            border-left: 4px solid #4a90e2;
        }
        
        .fusion-highlight {
            background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            border-left: 4px solid #7cb342;
        }
        
        .expert-info {
            background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            h1 {
                font-size: 2rem;
            }
            
            .card {
                padding: 15px;
            }
            
            .scores {
                flex-direction: column;
            }
            
            .score-item {
                margin: 5px 0;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🥤 自主现制饮品研发专家系统</h1>
            <p class="subtitle">真正的AI研发专家，从原料选择到工艺实现全流程自主完成（基于真实品牌营养补充剂）</p>
        </header>
        
        <div class="expert-info">
            <h3>🤖 专家系统能力</h3>
            <p>• 自主分析消费者需求并匹配营养成分（基于真实品牌数据）</p>
            <p>• 智能融合多种营养补充剂</p>
            <p>• 分析营养补充剂协同效应</p>
            <p>• 优化配方比例和成本控制</p>
            <p>• 生成详细制作工艺和质量控制方案</p>
            <p>• 提供市场分析和商业价值评估</p>
            <p>• 提供设备选型和操作参数建议</p>
        </div>
        
        <div class="card">
            <h2>🔬 智能饮品研发</h2>
            <div class="form-group">
                <label for="consumer-group">选择目标人群:</label>
                <select id="consumer-group">
                    <option value="上班族">上班族</option>
                    <option value="学生">学生</option>
                    <option value="中老年人">中老年人</option>
                    <option value="健身人群">健身人群</option>
                    <option value="爱美人士">爱美人士</option>
                </select>
            </div>
            <div class="form-group">
                <label for="health-goal">选择健康目标:</label>
                <select id="health-goal">
                    <option value="增强免疫力">增强免疫力</option>
                    <option value="增强记忆力">增强记忆力</option>
                    <option value="骨骼健康">骨骼健康</option>
                    <option value="肌肉增长">肌肉增长</option>
                    <option value="美容养颜">美容养颜</option>
                    <option value="抗氧化">抗氧化</option>
                    <option value="改善睡眠">改善睡眠</option>
                    <option value="抗疲劳">抗疲劳</option>
                </select>
            </div>
            <button onclick="generateFormulation()">启动AI研发专家</button>
            <div id="formulation-result"></div>
        </div>
    </div>

    <script>
        // 显示加载状态
        function showLoading(elementId) {
            document.getElementById(elementId).innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <p>AI研发专家正在自主研发中...</p>
                </div>
            `;
        }
        
        // 生成饮品配方
        function generateFormulation() {
            showLoading('formulation-result');
            
            const consumerGroup = document.getElementById('consumer-group').value;
            const healthGoal = document.getElementById('health-goal').value;
            
            fetch('/api/generate-formulation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    consumer_group: consumerGroup,
                    health_goal: healthGoal
                })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    document.getElementById('formulation-result').innerHTML = `
                        <div class="result">
                            <h3>❌ 错误</h3>
                            <p>${data.error}</p>
                        </div>
                    `;
                    return;
                }
                
                // 计算融合成分数量
                const nutrientCount = data.nutrients && data.nutrients.length > 0 ? data.nutrients.length : 0;
                const fusionLevel = nutrientCount > 3 ? "高" : nutrientCount > 1 ? "中" : "低";
                
                let html = `
                    <div class="result">
                        <h3>📋 研发成果报告</h3>
                        <p><strong>目标人群:</strong> ${data.consumer_group}</p>
                        <p><strong>健康目标:</strong> ${data.health_goal}</p>
                        <p><strong>基础饮品:</strong> ${data.base_beverage}</p>
                        
                        <div class="fusion-highlight">
                            <h4>🔄 多重营养补充剂融合</h4>
                            <p><strong>融合级别:</strong> ${fusionLevel}度融合 (${nutrientCount}种营养补充剂)</p>
                            <p>本配方智能融合了多种营养补充剂，实现协同增效效果</p>
                        </div>
                        
                        <h4 style="margin-top: 15px;">核心营养成分:</h4>
                        <ul class="nutrient-list">
                `;
                
                if (data.nutrients && data.nutrients.length > 0) {
                    data.nutrients.forEach((nutrient, index) => {
                        html += `
                            <li class="nutrient-item">
                                <div class="nutrient-name">${index + 1}. ${nutrient[0]} (${nutrient[1]}g)</div>
                            </li>
                        `;
                    });
                } else {
                    html += `<li class="nutrient-item">无特定营养成分</li>`;
                }
                
                html += `
                        </ul>
                        
                        <div class="synergy-analysis">
                            <h4>🔄 营养补充剂协同效应分析</h4>
                            <p><strong>协同效应总分:</strong> ${(data.synergy_analysis.synergy_score * 100).toFixed(1)}分</p>
                            <p><strong>成分兼容性:</strong> ${(data.synergy_analysis.compatibility_score * 100).toFixed(1)}%</p>
                            <p><strong>品牌协同度:</strong> ${(data.synergy_analysis.brand_synergy * 100).toFixed(1)}%</p>
                            <p><strong>类别协同度:</strong> ${(data.synergy_analysis.category_synergy * 100).toFixed(1)}%</p>
                            <p><strong>功效重叠数:</strong> ${data.synergy_analysis.health_benefit_overlap}项</p>
                            <details>
                                <summary style="cursor: pointer; font-weight: bold;">查看详细分析报告</summary>
                                <pre style="background: #f0f8ff; padding: 10px; margin-top: 10px; white-space: pre-wrap;">${data.synergy_analysis.detailed_analysis}</pre>
                            </details>
                        </div>
                        
                        <h4 style="margin-top: 15px;">口感调节剂:</h4>
                        <ul class="nutrient-list">
                `;
                
                if (data.flavor_adjusters && data.flavor_adjusters.length > 0) {
                    data.flavor_adjusters.forEach((adjuster, index) => {
                        html += `
                            <li class="nutrient-item">
                                <div class="nutrient-name">${index + 1}. ${adjuster[0]} (${adjuster[1]}g)</div>
                            </li>
                        `;
                    });
                } else {
                    html += `<li class="nutrient-item">无特定调节剂</li>`;
                }
                
                html += `
                        </ul>
                        
                        <div class="scores">
                            <div class="score-item">
                                <div>健康评分</div>
                                <div class="score-value">${(data.health_score * 100).toFixed(1)}%</div>
                            </div>
                            <div class="score-item">
                                <div>口感评分</div>
                                <div class="score-value">${(data.taste_score * 100).toFixed(1)}%</div>
                            </div>
                            <div class="score-item">
                                <div>成本评分</div>
                                <div class="score-value">${(data.cost_score * 100).toFixed(1)}%</div>
                            </div>
                            <div class="score-item">
                                <div>创新评分</div>
                                <div class="score-value">${(data.innovation_score * 100).toFixed(1)}%</div>
                            </div>
                        </div>
                        
                        <div class="market-analysis">
                            <h4>📊 市场分析</h4>
                            <p><strong>市场潜力指数:</strong> ${data.market_potential.toFixed(2)}</p>
                            <p><strong>建议售价:</strong> ¥${data.suggested_price.toFixed(2)}</p>
                            <p><strong>竞争优势指数:</strong> ${data.competitive_advantage.toFixed(2)}</p>
                        </div>
                        
                        <h4 style="margin-top: 15px;">制作工艺:</h4>
                        <div class="preparation-steps">${data.preparation_method}</div>
                    </div>
                `;
                
                document.getElementById('formulation-result').innerHTML = html;
            })
            .catch(error => {
                document.getElementById('formulation-result').innerHTML = `
                    <div class="result">
                        <h3>❌ 错误</h3>
                        <p>请求失败: ${error.message}</p>
                    </div>
                `;
            });
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    """主页"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/generate-formulation', methods=['POST'])
def generate_formulation():
    """生成饮品配方"""
    try:
        data = request.get_json()
        consumer_group = data.get('consumer_group', '上班族')
        health_goal = data.get('health_goal', '增强免疫力')
        
        result = expert_system.formulate_beverage(consumer_group, health_goal)
        
        # 转换为可JSON序列化的格式
        response_data = {
            'consumer_group': result.consumer_group,
            'health_goal': result.health_goal,
            'base_beverage': result.base_beverage,
            'nutrients': result.nutrients,
            'flavor_adjusters': result.flavor_adjusters,
            'preparation_method': result.preparation_method,
            'health_score': float(result.health_score),
            'taste_score': float(result.taste_score),
            'cost_score': float(result.cost_score),
            'innovation_score': float(result.innovation_score),
            'market_potential': float(result.market_potential),
            'suggested_price': float(result.suggested_price),
            'competitive_advantage': float(result.competitive_advantage),
            'synergy_analysis': {
                'synergy_score': float(result.synergy_analysis.synergy_score),
                'compatibility_score': float(result.synergy_analysis.compatibility_score),
                'brand_synergy': float(result.synergy_analysis.brand_synergy),
                'category_synergy': float(result.synergy_analysis.category_synergy),
                'health_benefit_overlap': int(result.synergy_analysis.health_benefit_overlap),
                'detailed_analysis': result.synergy_analysis.detailed_analysis
            }
        }
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def main():
    """主函数"""
    print("🚀 自主现制饮品研发专家系统（基于真实品牌营养补充剂）")
    print("=" * 60)
    print("系统特点:")
    print("✅ 真正的AI研发专家，具备自主决策能力")
    print("✅ 从原料选择到工艺实现全流程自主完成")
    print("✅ 基于深度学习的智能配方优化")
    print("✅ 提供完整的制作工艺和质量控制方案")
    print("✅ 集成市场分析和商业价值评估")
    print("✅ 支持多种营养补充剂智能融合")
    print("✅ 分析营养补充剂协同效应")
    print("✅ 基于真实品牌营养补充剂数据的专业研发")
    
    print("\n🔧 启动参数:")
    print("   访问地址: http://localhost:5000")
    print("   API端点:")
    print("     GET  / - Web界面")
    print("     POST /api/generate-formulation - 启动AI研发")
    
    print("\n💡 使用说明:")
    print("   1. 访问 http://localhost:5000 查看Web界面")
    print("   2. 选择目标人群和健康目标")
    print("   3. 点击启动AI研发专家")
    print("   4. 查看完整的研发成果报告")
    
    print("\n按 Ctrl+C 停止服务")
    
    try:
        app.run(host='localhost', port=5000, debug=False, use_reloader=False)
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except Exception as e:
        print(f"\n❌ 服务启动失败: {e}")

if __name__ == '__main__':
    main()