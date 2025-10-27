"""
基于强化学习的增强版自主现制饮品研发专家系统Web UI
支持更广泛的品牌、产品和人群数据
真正的AI研发专家，具备自主学习和决策能力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

# 导入增强版强化学习专家系统
from enhanced_rl_autonomous_beverage_expert import EnhancedRLAutonomousBeverageExpert

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 创建专家系统实例
expert_system = EnhancedRLAutonomousBeverageExpert()

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
    <title>增强版基于强化学习的自主现制饮品研发专家系统</title>
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
            flex-wrap: wrap;
        }
        
        .score-item {
            text-align: center;
            padding: 10px;
            background: #e3f2fd;
            border-radius: 8px;
            flex: 1;
            margin: 5px;
            min-width: 120px;
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
        
        .rl-analysis {
            background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            border-left: 4px solid #4a90e2;
        }
        
        .synergy-analysis {
            background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            border-left: 4px solid #7cb342;
        }
        
        .market-analysis {
            background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
            border-radius: 8px;
            padding: 15px;
            margin: 15px 0;
            border-left: 4px solid #f57c00;
        }
        
        .expert-info {
            background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
        
        .dataset-info {
            background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
            color: #333;
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
            <h1>🥤 增强版基于强化学习的自主现制饮品研发专家系统</h1>
            <p class="subtitle">真正的AI研发专家，具备自主学习和决策能力</p>
        </header>
        
        <div class="expert-info">
            <h3>🤖 专家系统能力</h3>
            <p>• 基于深度强化学习的智能决策</p>
            <p>• 自主分析消费者需求并匹配营养成分</p>
            <p>• 智能融合多种营养补充剂</p>
            <p>• 分析营养补充剂协同效应</p>
            <p>• 优化配方比例和成本控制</p>
            <p>• 生成详细制作工艺和质量控制方案</p>
            <p>• 持续学习和优化决策策略</p>
        </div>
        
        <div class="dataset-info">
            <h3>📊 数据集信息</h3>
            <p>• 支持28种知名品牌营养补充剂</p>
            <p>• 包含10类基础饮品载体</p>
            <p>• 提供10类口感调节剂</p>
            <p>• 覆盖10类目标消费群体</p>
            <p>• 实现成分兼容性智能分析</p>
            <p>• 提供精准的成本计算功能</p>
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
                    <option value="儿童青少年">儿童青少年</option>
                    <option value="孕产妇">孕产妇</option>
                    <option value="亚健康人群">亚健康人群</option>
                    <option value="减肥人群">减肥人群</option>
                    <option value="失眠人群">失眠人群</option>
                </select>
            </div>
            <div class="form-group">
                <label for="health-goal">选择健康目标:</label>
                <select id="health-goal">
                    <option value="增强免疫力">增强免疫力</option>
                    <option value="抗氧化">抗氧化</option>
                    <option value="骨骼健康">骨骼健康</option>
                    <option value="肌肉增长">肌肉增长</option>
                    <option value="美容养颜">美容养颜</option>
                    <option value="改善睡眠">改善睡眠</option>
                    <option value="抗疲劳">抗疲劳</option>
                    <option value="心血管保护">心血管保护</option>
                    <option value="肠道健康">肠道健康</option>
                    <option value="免疫调节">免疫调节</option>
                    <option value="营养补充">营养补充</option>
                    <option value="体力恢复">体力恢复</option>
                    <option value="促进胶原蛋白合成">促进胶原蛋白合成</option>
                    <option value="抗炎">抗炎</option>
                    <option value="抗压力">抗压力</option>
                    <option value="神经系统健康">神经系统健康</option>
                    <option value="能量代谢">能量代谢</option>
                    <option value="皮肤健康">皮肤健康</option>
                    <option value="头发强韧">头发强韧</option>
                    <option value="关节保护">关节保护</option>
                    <option value="滋阴">滋阴</option>
                    <option value="补血">补血</option>
                    <option value="改善便秘">改善便秘</option>
                    <option value="排毒">排毒</option>
                    <option value="延缓衰老">延缓衰老</option>
                    <option value="调节生物钟">调节生物钟</option>
                    <option value="减肥">减肥</option>
                    <option value="情绪调节">情绪调节</option>
                    <option value="认知功能">认知功能</option>
                    <option value="调理月经">调理月经</option>
                    <option value="护肝">护肝</option>
                    <option value="抗菌消炎">抗菌消炎</option>
                    <option value="口腔健康">口腔健康</option>
                    <option value="增强体质">增强体质</option>
                    <option value="滋阴补肾">滋阴补肾</option>
                    <option value="前列腺健康">前列腺健康</option>
                    <option value="全面营养">全面营养</option>
                    <option value="免疫支持">免疫支持</option>
                    <option value="肌肉功能">肌肉功能</option>
                    <option value="肌肉维护">肌肉维护</option>
                    <option value="牙齿健康">牙齿健康</option>
                    <option value="神经系统">神经系统</option>
                    <option value="脑部发育">脑部发育</option>
                    <option value="营养吸收">营养吸收</option>
                    <option value="生长发育">生长发育</option>
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
                    <p style="font-size: 0.9rem; margin-top: 10px;">基于强化学习的智能决策过程</p>
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
                const fusionLevel = nutrientCount > 5 ? "高" : nutrientCount > 3 ? "中" : "低";
                
                let html = `
                    <div class="result">
                        <h3>📋 研发成果报告</h3>
                        <p><strong>目标人群:</strong> ${data.consumer_group}</p>
                        <p><strong>健康目标:</strong> ${data.health_goal}</p>
                        <p><strong>基础饮品:</strong> ${data.base_beverage}</p>
                        
                        <div class="rl-analysis">
                            <h4>🤖 强化学习决策分析</h4>
                            <p><strong>决策策略:</strong> 基于深度Q网络的智能决策</p>
                            <p><strong>探索率:</strong> ${data.rl_metrics.epsilon.toFixed(3)}</p>
                            <p><strong>经验回放:</strong> ${data.rl_metrics.experience_size} 条经验</p>
                            <p><strong>平均性能:</strong> ${data.rl_metrics.avg_performance.toFixed(2)}</p>
                        </div>
                        
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
                            <div class="score-item">
                                <div>市场潜力</div>
                                <div class="score-value">${(data.market_potential * 100).toFixed(1)}%</div>
                            </div>
                        </div>
                        
                        <div class="market-analysis">
                            <h4>📈 市场价值分析</h4>
                            <p><strong>建议售价:</strong> ¥${data.suggested_price.toFixed(2)}</p>
                            <p><strong>竞争优势:</strong> ${(data.competitive_advantage * 100).toFixed(1)}分</p>
                            <p><strong>目标消费群体:</strong> ${data.consumer_group}</p>
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
        
        # 计算建议售价（基于成本和市场定位）
        base_cost = 0.0
        # 基础饮品成本
        base_beverage_info = expert_system.base_beverages.get(result.base_beverage, {})
        base_cost += base_beverage_info.get("price_per_liter", 0) * 0.5  # 500ml
        
        # 营养成分成本
        for nutrient_name, amount in result.nutrients:
            nutrient_info = expert_system.nutrient_database.get(nutrient_name)
            if nutrient_info:
                ingredient_cost = nutrient_info.price_per_kg * (amount / 1000) * 0.5
                base_cost += ingredient_cost
        
        # 调味剂成本
        for adjuster_name, amount in result.flavor_adjusters:
            adjuster_info = expert_system.flavor_adjusters.get(adjuster_name)
            if adjuster_info:
                price_per_kg = adjuster_info.get("price_per_gram", 0) * 1000
                adjuster_cost = price_per_kg * (amount / 1000) * 0.5
                base_cost += adjuster_cost
        
        # 建议售价 = 成本 * 5 + 市场潜力调整
        suggested_price = base_cost * 5 + (result.market_potential * 10)
        # 竞争优势指数
        competitive_advantage = (result.health_score * 0.4 + result.taste_score * 0.3 + 
                               result.innovation_score * 0.2 + result.market_potential * 0.1)
        
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
            'suggested_price': float(suggested_price),
            'competitive_advantage': float(competitive_advantage),
            'synergy_analysis': {
                'synergy_score': float(result.synergy_analysis.synergy_score),
                'compatibility_score': float(result.synergy_analysis.compatibility_score),
                'brand_synergy': float(result.synergy_analysis.brand_synergy),
                'category_synergy': float(result.synergy_analysis.category_synergy),
                'health_benefit_overlap': int(result.synergy_analysis.health_benefit_overlap),
                'detailed_analysis': result.synergy_analysis.detailed_analysis
            },
            'rl_metrics': {
                'epsilon': float(expert_system.rl_agent.epsilon),
                'experience_size': len(expert_system.rl_agent.memory),
                'avg_performance': float(np.mean(expert_system.performance_history)) if expert_system.performance_history else 0.0
            }
        }
        
        return jsonify(response_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def main():
    """主函数"""
    print("🚀 增强版基于强化学习的自主现制饮品研发专家系统")
    print("=" * 60)
    print("系统特点:")
    print("✅ 真正的AI研发专家，具备自主学习和决策能力")
    print("✅ 基于深度强化学习的智能决策")
    print("✅ 从原料选择到工艺实现全流程自主完成")
    print("✅ 提供完整的制作工艺和质量控制方案")
    print("✅ 支持多种营养补充剂智能融合")
    print("✅ 分析营养补充剂协同效应")
    print("✅ 支持更广泛的品牌、产品和人群")
    
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