"""
基于强化学习的自主现制饮品研发专家系统Web UI
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

# 导入强化学习专家系统
from rl_autonomous_beverage_expert import RLAutonomousBeverageExpert

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 创建专家系统实例
expert_system = RLAutonomousBeverageExpert()

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
    <title>基于强化学习的自主现制饮品研发专家系统</title>
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
        
        .tabs {
            display: flex;
            margin-bottom: 20px;
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .tab-button {
            flex: 1;
            padding: 15px;
            text-align: center;
            background: #f8f9fa;
            border: none;
            cursor: pointer;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        
        .tab-button:hover {
            background: #e9ecef;
        }
        
        .tab-button.active {
            background: #667eea;
            color: white;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
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
        
        .expert-info {
            background: linear-gradient(135deg, #ffd89b 0%, #19547b 100%);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin: 20px 0;
        }
        
        /* 架构可视化样式 */
        .architecture-overview {
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin-bottom: 30px;
        }
        
        .architecture-component {
            flex: 1;
            min-width: 250px;
            background: white;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .architecture-component:hover {
            transform: scale(1.03);
        }
        
        .component-icon {
            font-size: 3rem;
            margin-bottom: 15px;
        }
        
        .component-name {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 10px;
        }
        
        .component-role {
            font-size: 1rem;
            color: #666;
            margin-bottom: 15px;
            font-style: italic;
        }
        
        .component-description {
            font-size: 0.95rem;
            color: #555;
        }
        
        .workflow-container {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
            margin-bottom: 30px;
        }
        
        .workflow-title {
            text-align: center;
            font-size: 1.8rem;
            margin-bottom: 25px;
            color: #333;
        }
        
        .workflow-steps {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }
        
        .workflow-step {
            display: flex;
            align-items: center;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            transition: all 0.3s ease;
        }
        
        .workflow-step.active {
            background: #e3f2fd;
            transform: scale(1.02);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .step-number {
            width: 40px;
            height: 40px;
            background: #667eea;
            color: white;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            margin-right: 20px;
            flex-shrink: 0;
        }
        
        .step-content {
            flex: 1;
        }
        
        .step-title {
            font-size: 1.2rem;
            font-weight: 600;
            margin-bottom: 5px;
        }
        
        .step-description {
            color: #666;
            margin-bottom: 5px;
        }
        
        .step-time {
            font-size: 0.9rem;
            color: #888;
            font-style: italic;
        }
        
        .active-architectures {
            display: flex;
            gap: 10px;
            margin-top: 10px;
        }
        
        .active-architecture-tag {
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 0.8rem;
            font-weight: 500;
        }
        
        .mechanism-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }
        
        .mechanism-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 5px 20px rgba(0,0,0,0.1);
        }
        
        .mechanism-title {
            font-size: 1.3rem;
            margin-bottom: 15px;
            color: #333;
        }
        
        .mechanism-description {
            color: #666;
            margin-bottom: 15px;
            line-height: 1.6;
        }
        
        .mechanism-example {
            background: #e3f2fd;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            font-style: italic;
        }
        
        .benefits-list {
            list-style-type: none;
        }
        
        .benefits-list li {
            padding: 8px 0;
            border-bottom: 1px solid #f0f0f0;
        }
        
        .benefits-list li:last-child {
            border-bottom: none;
        }
        
        .benefits-list li:before {
            content: "✓";
            color: #4CAF50;
            font-weight: bold;
            margin-right: 10px;
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
            
            .architecture-overview {
                flex-direction: column;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>🥤 基于强化学习的自主现制饮品研发专家系统</h1>
            <p class="subtitle">真正的AI研发专家，具备自主学习和决策能力</p>
        </header>
        
        <div class="tabs">
            <button class="tab-button active" onclick="switchTab('formulation')">智能饮品研发</button>
            <button class="tab-button" onclick="switchTab('architecture')">AI架构可视化</button>
        </div>
        
        <div id="formulation" class="tab-content active">
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
        
        <div id="architecture" class="tab-content">
            <div class="card">
                <h2>🧠 AI系统架构可视化</h2>
                <p>本系统采用多架构融合设计，包含三个主要组成部分，协同工作实现智能饮品研发。</p>
                
                <div class="architecture-overview">
                    <div class="architecture-component" style="border-top: 5px solid #FF6B6B;">
                        <div class="component-icon">🧠</div>
                        <div class="component-name">强化学习主架构</div>
                        <div class="component-role">决策指挥官</div>
                        <div class="component-description">制定整体研发策略，控制研发流程，优化长期目标，平衡探索与利用</div>
                    </div>
                    
                    <div class="architecture-component" style="border-top: 5px solid #4ECDC4;">
                        <div class="component-icon">🤖</div>
                        <div class="component-name">Transformer专用架构</div>
                        <div class="component-role">关系分析师</div>
                        <div class="component-description">分析成分间复杂关系，识别协同效应，预测成分相互作用，提供关系洞察</div>
                    </div>
                    
                    <div class="architecture-component" style="border-top: 5px solid #45B7D1;">
                        <div class="component-icon">🔬</div>
                        <div class="component-name">专用多模型架构</div>
                        <div class="component-role">专业执行者</div>
                        <div class="component-description">成分选择推荐，配方比例优化，质量综合评估，领域知识应用</div>
                    </div>
                </div>
                
                <div class="workflow-container">
                    <h3 class="workflow-title">🚀 研发工作流程</h3>
                    <div class="workflow-steps">
                        <div class="workflow-step">
                            <div class="step-number">1</div>
                            <div class="step-content">
                                <div class="step-title">用户需求输入</div>
                                <div class="step-description">用户输入健康目标和个人信息</div>
                                <div class="step-time">预计时间: 0秒</div>
                            </div>
                        </div>
                        
                        <div class="workflow-step">
                            <div class="step-number">2</div>
                            <div class="step-content">
                                <div class="step-title">强化学习制定策略</div>
                                <div class="step-description">强化学习架构分析用户需求，制定研发策略</div>
                                <div class="step-time">预计时间: 2秒</div>
                                <div class="active-architectures">
                                    <span class="active-architecture-tag" style="background-color: #FF6B6B20; color: #FF6B6B; border: 1px solid #FF6B6B;">强化学习架构</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="workflow-step">
                            <div class="step-number">3</div>
                            <div class="step-content">
                                <div class="step-title">成分选择推荐</div>
                                <div class="step-description">专用模型架构推荐合适的营养成分</div>
                                <div class="step-time">预计时间: 1秒</div>
                                <div class="active-architectures">
                                    <span class="active-architecture-tag" style="background-color: #45B7D120; color: #45B7D1; border: 1px solid #45B7D1;">专用模型架构</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="workflow-step">
                            <div class="step-number">4</div>
                            <div class="step-content">
                                <div class="step-title">成分关系分析</div>
                                <div class="step-description">Transformer架构分析成分间协同效应</div>
                                <div class="step-time">预计时间: 1.5秒</div>
                                <div class="active-architectures">
                                    <span class="active-architecture-tag" style="background-color: #4ECDC420; color: #4ECDC4; border: 1px solid #4ECDC4;">Transformer架构</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="workflow-step">
                            <div class="step-number">5</div>
                            <div class="step-content">
                                <div class="step-title">配方优化</div>
                                <div class="step-description">专用模型架构优化成分比例</div>
                                <div class="step-time">预计时间: 1秒</div>
                                <div class="active-architectures">
                                    <span class="active-architecture-tag" style="background-color: #45B7D120; color: #45B7D1; border: 1px solid #45B7D1;">专用模型架构</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="workflow-step">
                            <div class="step-number">6</div>
                            <div class="step-content">
                                <div class="step-title">质量评估</div>
                                <div class="step-description">专用模型架构评估配方质量</div>
                                <div class="step-time">预计时间: 1秒</div>
                                <div class="active-architectures">
                                    <span class="active-architecture-tag" style="background-color: #45B7D120; color: #45B7D1; border: 1px solid #45B7D1;">专用模型架构</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="workflow-step">
                            <div class="step-number">7</div>
                            <div class="step-content">
                                <div class="step-title">策略学习优化</div>
                                <div class="step-description">强化学习架构根据结果优化策略</div>
                                <div class="step-time">预计时间: 0.5秒</div>
                                <div class="active-architectures">
                                    <span class="active-architecture-tag" style="background-color: #FF6B6B20; color: #FF6B6B; border: 1px solid #FF6B6B;">强化学习架构</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="workflow-step">
                            <div class="step-number">8</div>
                            <div class="step-content">
                                <div class="step-title">最终配方输出</div>
                                <div class="step-description">输出个性化功能性饮品配方</div>
                                <div class="step-time">预计时间: 0秒</div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="mechanism-cards">
                    <div class="mechanism-card">
                        <h3 class="mechanism-title">🔄 架构间通信机制</h3>
                        <div class="mechanism-description">确保各架构高效协作</div>
                        
                        <h4>异步通信机制</h4>
                        <div class="mechanism-example">就像三个厨师同时准备不同菜品</div>
                        <p>各架构并行工作提高效率，不需要等待其他架构完成任务。</p>
                        <ul class="benefits-list">
                            <li>速度快：多个架构同时工作</li>
                            <li>资源利用好：充分利用计算资源</li>
                            <li>响应及时：用户能快速看到部分结果</li>
                        </ul>
                        
                        <h4>共享内存机制</h4>
                        <div class="mechanism-example">就像办公室公告板，大家都能看到信息</div>
                        <p>架构间数据共享和通信，确保信息透明。</p>
                        <ul class="benefits-list">
                            <li>信息透明：每个架构都能看到其他架构的工作成果</li>
                            <li>避免重复工作：不需要每个架构都重新计算相同的信息</li>
                            <li>决策更全面：可以基于所有架构的结果做出更好决策</li>
                        </ul>
                    </div>
                    
                    <div class="mechanism-card">
                        <h3 class="mechanism-title">⚙️ 动态协调机制</h3>
                        <div class="mechanism-description">智能分配任务并持续优化</div>
                        
                        <h4>任务分配与调度</h4>
                        <div class="mechanism-example">就像医院分诊，不同病情找不同科室</div>
                        <p>根据任务特点分配给最合适的专家。</p>
                        <ul class="benefits-list">
                            <li>制定研发策略 → 强化学习架构</li>
                            <li>分析成分关系 → Transformer架构</li>
                            <li>推荐具体成分 → 专用模型架构</li>
                            <li>计算成分剂量 → 专用模型架构</li>
                            <li>评估配方质量 → 专用模型架构</li>
                        </ul>
                        
                        <h4>反馈与学习机制</h4>
                        <div class="mechanism-example">就像厨师根据客人反馈调整菜品口味</div>
                        <p>根据用户满意度调整各架构工作方式。</p>
                        <ul class="benefits-list">
                            <li>收集反馈：记录用户对配方的满意度</li>
                            <li>调整参数：根据反馈优化各架构性能</li>
                            <li>持续优化：不断改进系统表现</li>
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // 标签页切换功能
        function switchTab(tabName) {
            // 隐藏所有标签内容
            document.querySelectorAll('.tab-content').forEach(tab => {
                tab.classList.remove('active');
            });
            
            // 移除所有标签按钮的激活状态
            document.querySelectorAll('.tab-button').forEach(button => {
                button.classList.remove('active');
            });
            
            // 显示选中的标签内容
            document.getElementById(tabName).classList.add('active');
            
            // 激活选中的标签按钮
            event.target.classList.add('active');
        }
        
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
                const fusionLevel = nutrientCount > 3 ? "高" : nutrientCount > 1 ? "中" : "低";
                
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
    print("🚀 基于强化学习的自主现制饮品研发专家系统")
    print("=" * 60)
    print("系统特点:")
    print("✅ 真正的AI研发专家，具备自主学习和决策能力")
    print("✅ 基于深度强化学习的智能决策")
    print("✅ 从原料选择到工艺实现全流程自主完成")
    print("✅ 提供完整的制作工艺和质量控制方案")
    print("✅ 支持多种营养补充剂智能融合")
    print("✅ 分析营养补充剂协同效应")
    print("✅ 可视化展示AI架构工作原理")
    
    print("\n🔧 启动参数:")
    print("   访问地址: http://localhost:5000")
    print("   API端点:")
    print("     GET  / - Web界面")
    print("     POST /api/generate-formulation - 启动AI研发")
    
    print("\n💡 使用说明:")
    print("   1. 访问 http://localhost:5000 查看Web界面")
    print("   2. 使用'智能饮品研发'标签进行配方研发")
    print("   3. 切换到'AI架构可视化'标签了解系统工作原理")
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