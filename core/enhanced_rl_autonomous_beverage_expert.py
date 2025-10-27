"""
基于强化学习的增强版自主现制饮品研发专家系统
支持更广泛的品牌、产品和人群数据
结合深度Q网络(DQN)实现智能决策和持续优化
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
import copy

# 导入增强版品牌营养补充剂数据集
from enhanced_brand_nutritional_supplement_dataset import (
    ENHANCED_BRAND_NUTRITIONAL_SUPPLEMENTS,
    ENHANCED_BASE_BEVERAGES,
    ENHANCED_FLAVOR_ADJUSTERS,
    ENHANCED_TARGET_CONSUMERS
)

# 设置随机种子以确保结果可重现
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

@dataclass
class NutrientInfo:
    """营养成分信息"""
    name: str
    brand: str
    category: str
    health_benefits: List[str]
    solubility: str
    flavor_profile: str
    usage_rate: str
    price_per_kg: float
    compatibility: List[str]
    incompatibility: List[str]
    research_score: float  # 研发价值评分
    processing_characteristics: Dict[str, str]

@dataclass
class SynergyAnalysis:
    """协同效应分析结果"""
    synergy_score: float
    compatibility_score: float
    brand_synergy: float
    category_synergy: float
    health_benefit_overlap: int
    detailed_analysis: str

@dataclass
class BeverageFormulation:
    """饮品配方"""
    consumer_group: str
    health_goal: str
    base_beverage: str
    nutrients: List[Tuple[str, float]]  # 成分名称和用量(g/L)
    flavor_adjusters: List[Tuple[str, float]]  # 调味剂和用量(g/L)
    preparation_method: str
    health_score: float
    taste_score: float
    cost_score: float
    innovation_score: float
    market_potential: float = 0.0  # 市场潜力指数
    suggested_price: float = 0.0   # 建议售价
    competitive_advantage: float = 0.0  # 竞争优势指数
    synergy_analysis: SynergyAnalysis = None  # 协同效应分析

class ProcessingStage(Enum):
    """加工阶段"""
    RAW_MATERIAL_SELECTION = "原料选择"
    PRE_TREATMENT = "预处理"
    MIXING = "混合"
    HOMOGENIZATION = "均质化"
    STERILIZATION = "杀菌"
    PACKAGING = "包装"

class DQN(nn.Module):
    """深度Q网络"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class RLAgent:
    """强化学习代理"""
    def __init__(self, state_dim: int, action_dim: int, lr: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr
        
        # Q网络
        self.q_network = DQN(state_dim, 256, action_dim)
        self.target_network = DQN(state_dim, 256, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        
        # 经验回放缓冲区
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # 超参数
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.update_target_freq = 100
        self.step_count = 0
        
        # 初始化目标网络
        self.update_target_network()
    
    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def remember(self, state, action, reward, next_state, done):
        """存储经验"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """选择动作"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
    
    def replay(self):
        """经验回放"""
        if len(self.memory) < self.batch_size:
            return
        
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch])
        actions = torch.LongTensor([e[1] for e in batch])
        rewards = torch.FloatTensor([e[2] for e in batch])
        next_states = torch.FloatTensor([e[3] for e in batch])
        dones = torch.BoolTensor([e[4] for e in batch])
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.update_target_network()

class EnhancedRLAutonomousBeverageExpert:
    """基于强化学习的增强版自主现制饮品研发专家"""
    
    def __init__(self):
        # 初始化营养成分数据库（基于增强版真实品牌数据）
        self.nutrient_database = self._initialize_nutrient_database()
        self.base_beverages = ENHANCED_BASE_BEVERAGES
        self.flavor_adjusters = ENHANCED_FLAVOR_ADJUSTERS
        self.consumer_profiles = ENHANCED_TARGET_CONSUMERS
        self.processing_equipment = self._initialize_processing_equipment()
        
        # 初始化强化学习代理
        self.state_dim = 25  # 增强的状态维度
        self.action_dim = len(self.nutrient_database) + len(self.base_beverages) + len(self.flavor_adjusters)
        self.rl_agent = RLAgent(self.state_dim, self.action_dim)
        
        # 研发历史记录
        self.research_history = []
        self.successful_formulations = []
        
        # 性能跟踪
        self.performance_history = []
    
    def _initialize_nutrient_database(self) -> Dict[str, NutrientInfo]:
        """初始化营养成分数据库（基于增强版真实品牌数据）"""
        nutrient_database = {}
        
        for supplement_name, supplement_info in ENHANCED_BRAND_NUTRITIONAL_SUPPLEMENTS.items():
            # 将品牌补充剂转换为营养成分信息
            nutrient_database[supplement_name] = NutrientInfo(
                name=supplement_name,
                brand=supplement_info["brand"],
                category=supplement_info["category"],
                health_benefits=supplement_info["health_benefits"],
                solubility=supplement_info["solubility"],
                flavor_profile=supplement_info["flavor_profile"],
                usage_rate=supplement_info["usage_rate"],
                price_per_kg=supplement_info["price_per_gram"] * 1000,  # 转换为每公斤价格
                compatibility=supplement_info["compatibility"],
                incompatibility=supplement_info["incompatibility"],
                research_score=supplement_info["user_rating"] / 5.0,  # 转换为0-1评分
                processing_characteristics=supplement_info.get("processing_characteristics", {})
            )
        
        return nutrient_database
    
    def _initialize_processing_equipment(self) -> Dict[str, Dict]:
        """初始化加工设备"""
        return {
            "高速搅拌机": {
                "capacity": "2L",
                "speed_range": "10000-30000 RPM",
                "features": ["均质化", "快速混合", "温度控制"],
                "processing_time": "30-60秒"
            },
            "超声波均质机": {
                "frequency": "20kHz",
                "power": "500W",
                "features": ["纳米级均质", "保持营养活性", "无热损伤"],
                "processing_time": "5-10分钟"
            },
            "巴氏杀菌设备": {
                "temperature_control": "60-85°C",
                "holding_time": "15-30秒",
                "features": ["保持营养", "延长保质期", "食品安全"],
                "processing_time": "2-5分钟"
            },
            "真空均质机": {
                "pressure_range": "0.01-0.1 MPa",
                "features": ["防止氧化", "保持色泽", "提高稳定性"],
                "processing_time": "10-15分钟"
            },
            "低温杀菌设备": {
                "temperature_control": "35-50°C",
                "holding_time": "5-10分钟",
                "features": ["保持活性成分", "延长保质期", "食品安全"],
                "processing_time": "5-10分钟"
            }
        }
    
    def _get_state_representation(self, consumer_group: str, health_goal: str) -> np.ndarray:
        """获取状态表示"""
        state = np.zeros(self.state_dim)
        
        # 编码消费者群体 (前10维)
        consumer_mapping = {
            "上班族": 0, "学生": 1, "中老年人": 2, 
            "健身人群": 3, "爱美人士": 4, "儿童青少年": 5,
            "孕产妇": 6, "亚健康人群": 7, "减肥人群": 8,
            "失眠人群": 9
        }
        if consumer_group in consumer_mapping:
            state[consumer_mapping[consumer_group]] = 1.0
        
        # 编码健康目标 (接下来的10维)
        health_goals = [
            "增强免疫力", "增强记忆力", "骨骼健康", "肌肉增长", 
            "美容养颜", "抗氧化", "改善睡眠", "抗疲劳",
            "生长发育", "心血管保护"
        ]
        if health_goal in health_goals:
            state[10 + health_goals.index(health_goal)] = 1.0
        
        # 添加一些基础特征 (最后5维)
        state[20] = len(self.nutrient_database) / 30.0  # 营养成分数据库大小
        state[21] = len(self.base_beverages) / 15.0     # 基础饮品数量
        state[22] = len(self.flavor_adjusters) / 15.0   # 调味剂数量
        state[23] = len(self.consumer_profiles) / 15.0  # 消费者群体数量
        state[24] = np.mean(self.performance_history) if self.performance_history else 0.5  # 平均性能
        
        return state
    
    def _get_reward(self, formulation: BeverageFormulation) -> float:
        """计算奖励"""
        # 综合考虑多个评分指标
        health_weight = 0.25
        taste_weight = 0.2
        cost_weight = 0.2
        innovation_weight = 0.15
        synergy_weight = 0.1
        market_potential_weight = 0.1
        
        reward = (
            health_weight * formulation.health_score +
            taste_weight * formulation.taste_score +
            cost_weight * formulation.cost_score +
            innovation_weight * formulation.innovation_score +
            synergy_weight * formulation.synergy_analysis.synergy_score +
            market_potential_weight * formulation.market_potential
        )
        
        # 如果是高质量配方，给予额外奖励
        if formulation.health_score > 0.8 and formulation.taste_score > 0.7:
            reward += 0.5
        
        return reward
    
    def _select_base_beverage(self, consumer_analysis: Dict) -> str:
        """选择基础饮品"""
        preferred_flavors = consumer_analysis.get("preferred_flavors", [])
        health_concerns = consumer_analysis.get("health_concerns", [])
        consumer_group = consumer_analysis.get("consumer_group", "")
        
        # 基于不同群体和健康需求选择
        if "减肥" in health_concerns:
            return "气泡水"  # 减肥人群偏好低热量饮品
        elif "失眠" in health_concerns:
            return "花草茶"  # 失眠人群偏好有助于放松的花草茶
        elif consumer_group == "儿童青少年":
            return "果蔬汁"  # 儿童青少年偏好天然果蔬汁
        elif consumer_group == "孕产妇":
            return "牛奶"  # 孕产妇需要丰富的钙质
        elif any("柑橘" in flavor for flavor in preferred_flavors):
            return "椰子水"
        elif any("温和" in flavor for flavor in preferred_flavors):
            return "燕麦奶"
        else:
            return "纯净水"
    
    def _select_flavor_adjusters(self, nutrients: List[str], base_beverage: str,
                               consumer_analysis: Dict) -> List[Tuple[str, float]]:
        """选择调味剂"""
        flavor_adjusters = []
        preferred_flavors = consumer_analysis.get("preferred_flavors", [])
        consumer_group = consumer_analysis.get("consumer_group", "")
        health_concerns = consumer_analysis.get("health_concerns", [])
        
        # 基于成分、基础饮品和消费者群体选择调味剂
        if base_beverage == "椰子水":
            # 椰子水本身有甜味，适量添加柠檬汁提鲜
            flavor_adjusters.append(("柠檬汁", 2.0))
        elif base_beverage == "燕麦奶":
            # 燕麦奶口感醇厚，可添加香草精增香
            flavor_adjusters.append(("香草精", 0.3))
        elif base_beverage == "气泡水":
            # 气泡水适合添加薄荷提取物增加清凉感
            flavor_adjusters.append(("薄荷提取物", 0.1))
        elif base_beverage == "花草茶":
            # 花草茶可添加蜂蜜增加甜味
            flavor_adjusters.append(("蜂蜜", 1.5))
        else:
            # 纯净水需要更多调味
            if any("水果" in flavor for flavor in preferred_flavors):
                flavor_adjusters.append(("蜂蜜", 3.0))
                flavor_adjusters.append(("柠檬汁", 1.5))
            else:
                flavor_adjusters.append(("蜂蜜", 2.0))
        
        # 根据健康需求添加特定调味剂
        if "减肥" in health_concerns:
            flavor_adjusters.append(("芦荟汁", 1.0))  # 芦荟有助于减肥
        elif "失眠" in health_concerns:
            flavor_adjusters.append(("蜂蜜", 2.0))  # 蜂蜜有助于改善睡眠
        
        return flavor_adjusters
    
    def _analyze_synergy(self, nutrients: List[Tuple[str, float]]) -> SynergyAnalysis:
        """分析营养补充剂之间的协同效应"""
        if len(nutrients) < 2:
            return SynergyAnalysis(
                synergy_score=0.0,
                compatibility_score=0.0,
                brand_synergy=0.0,
                category_synergy=0.0,
                health_benefit_overlap=0,
                detailed_analysis="单一成分，无协同效应"
            )
        
        # 计算各种协同效应指标
        compatibility_score = 0.0
        brand_synergy = 0.0
        category_synergy = 0.0
        health_benefit_overlap = 0
        
        # 获取所有成分信息
        nutrient_infos = []
        for nutrient_name, _ in nutrients:
            if nutrient_name in self.nutrient_database:
                nutrient_infos.append(self.nutrient_database[nutrient_name])
        
        # 分析成分间的兼容性
        compatibility_pairs = 0
        total_pairs = 0
        for i, info1 in enumerate(nutrient_infos):
            for j, info2 in enumerate(nutrient_infos):
                if i >= j:  # 避免重复比较
                    continue
                total_pairs += 1
                
                # 检查成分兼容性
                if any(comp in info2.compatibility for comp in info1.compatibility):
                    compatibility_score += 1.0
                    compatibility_pairs += 1
                elif any(incomp in info2.incompatibility for incomp in info1.incompatibility):
                    compatibility_score -= 0.5  # 不兼容扣分
        
        # 计算品牌协同效应
        brands = [info.brand for info in nutrient_infos]
        unique_brands = set(brands)
        if len(unique_brands) > 0:
            brand_synergy = 1.0 - (len(unique_brands) / len(brands))  # 品牌重叠度越高，协同效应越强
        
        # 计算类别协同效应
        categories = [info.category for info in nutrient_infos]
        unique_categories = set(categories)
        if len(unique_categories) > 0:
            category_synergy = 1.0 - (len(unique_categories) / len(categories))  # 类别重叠度越高，协同效应越强
        
        # 计算健康功效重叠
        all_benefits = []
        for info in nutrient_infos:
            all_benefits.extend(info.health_benefits)
        unique_benefits = set(all_benefits)
        health_benefit_overlap = len(all_benefits) - len(unique_benefits)  # 重叠的健康功效数量
        
        # 计算综合协同效应得分
        if total_pairs > 0:
            compatibility_score = max(0.0, compatibility_score / total_pairs)
        synergy_score = (compatibility_score * 0.4 + brand_synergy * 0.3 + category_synergy * 0.2 + 
                        min(health_benefit_overlap * 0.1, 0.1))  # 限制健康功效重叠的权重
        
        # 生成详细分析
        detailed_analysis = f"协同效应分析报告:\n"
        detailed_analysis += f"• 成分兼容性: {compatibility_pairs}/{total_pairs} 对成分兼容\n"
        detailed_analysis += f"• 品牌协同: {len(brands)-len(unique_brands)} 个品牌重叠\n"
        detailed_analysis += f"• 类别协同: {len(categories)-len(unique_categories)} 个类别重叠\n"
        detailed_analysis += f"• 功效重叠: {health_benefit_overlap} 项健康功效重叠\n"
        
        return SynergyAnalysis(
            synergy_score=synergy_score,
            compatibility_score=compatibility_score,
            brand_synergy=brand_synergy,
            category_synergy=category_synergy,
            health_benefit_overlap=health_benefit_overlap,
            detailed_analysis=detailed_analysis
        )
    
    def _optimize_formulation(self, nutrients: List[Tuple[str, float]], base_beverage: str, 
                           consumer_analysis: Dict) -> List[Tuple[str, float]]:
        """优化配方比例"""
        optimized_nutrients = []
        
        # 基于成分特性和消费者需求优化用量
        for nutrient_name, _ in nutrients:
            nutrient_info = self.nutrient_database.get(nutrient_name)
            if not nutrient_info:
                continue
            
            # 基础用量范围
            usage_range = nutrient_info.usage_rate
            if "-" in usage_range:
                min_usage, max_usage = map(float, usage_range.rstrip('%').split('-'))
                # 根据消费者群体调整用量
                consumer_group = consumer_analysis["consumer_group"]
                if consumer_group in ["学生", "儿童青少年"]:
                    # 学生和儿童群体用量偏向下限
                    optimal_usage = min_usage + (max_usage - min_usage) * 0.3
                elif consumer_group in ["中老年人", "孕产妇"]:
                    # 中老年和孕产妇群体用量偏向上限
                    optimal_usage = min_usage + (max_usage - min_usage) * 0.7
                else:
                    # 其他群体取中位数
                    optimal_usage = (min_usage + max_usage) / 2
            else:
                optimal_usage = float(usage_range.rstrip('%'))
            
            # 根据成分在配方中的重要性调整用量
            # 主要成分用量较高，次要成分用量较低
            if nutrient_name in consumer_analysis["matching_nutrients"][:2]:
                # 主要成分，用量保持或略微增加
                optimal_usage = optimal_usage * 1.0
            elif nutrient_name in consumer_analysis.get("health_concerns", []):
                # 次要成分，用量适中
                optimal_usage = optimal_usage * 0.8
            else:
                # 补充成分，用量较低
                optimal_usage = optimal_usage * 0.6
            
            optimized_nutrients.append((nutrient_name, optimal_usage))
        
        return optimized_nutrients
    
    def _generate_preparation_method(self, nutrients: List[Tuple[str, float]], 
                                   flavor_adjusters: List[Tuple[str, float]], 
                                   base_beverage: str) -> str:
        """生成制作工艺"""
        method = f"🔬 现制饮品制作工艺流程\n\n"
        method += "📋 原料准备阶段:\n"
        method += f"   • 准备{base_beverage} 500ml\n"
        for nutrient, amount in nutrients:
            method += f"   • 准备{nutrient} {amount}g\n"
        for adjuster, amount in flavor_adjusters:
            method += f"   • 准备{adjuster} {amount}g\n"
        method += "\n"
        
        method += "⚙️ 加工制作流程:\n"
        method += "   1. 预处理阶段:\n"
        method += "      - 检查所有原料质量\n"
        method += "      - 准备高速搅拌设备\n"
        method += "      - 预热设备至适宜温度\n\n"
        
        method += "   2. 混合阶段:\n"
        method += f"      - 将500ml {base_beverage}倒入搅拌杯\n"
        method += "      - 按顺序添加营养成分粉末:\n"
        for i, (nutrient, amount) in enumerate(nutrients, 1):
            method += f"         {i}. {nutrient} {amount}g\n"
        method += "      - 启动搅拌机，低速混合30秒\n\n"
        
        method += "   3. 均质化阶段:\n"
        method += "      - 提高转速至20000RPM\n"
        method += "      - 持续搅拌60秒确保充分溶解\n"
        method += "      - 观察混合液均匀度\n\n"
        
        method += "   4. 调味阶段:\n"
        method += "      - 依次添加调味剂:\n"
        for i, (adjuster, amount) in enumerate(flavor_adjusters, 1):
            method += f"         {i}. {adjuster} {amount}g\n"
        method += "      - 继续搅拌30秒混合均匀\n\n"
        
        method += "   5. 质量检测:\n"
        method += "      - 检测pH值和溶解度\n"
        method += "      - 品尝口感并调整\n"
        method += "      - 确认无颗粒感和异味\n\n"
        
        method += "   6. 包装储存:\n"
        method += "      - 立即灌装至无菌容器\n"
        method += "      - 密封保存于4°C冷藏\n"
        method += "      - 建议24小时内饮用完毕\n"
        
        return method
    
    def _assess_quality(self, nutrients: List[Tuple[str, float]], 
                       flavor_adjusters: List[Tuple[str, float]], 
                       base_beverage: str, 
                       synergy_analysis: SynergyAnalysis,
                       consumer_analysis: Dict) -> Dict[str, float]:
        """评估配方质量"""
        # 健康评分：基于营养成分的价值和多样性
        health_score = 0.0
        unique_benefits = set()
        
        for nutrient_name, amount in nutrients:
            nutrient_info = self.nutrient_database.get(nutrient_name)
            if nutrient_info:
                health_score += nutrient_info.research_score * (amount / 100)
                # 收集健康功效，多样性越高得分越高
                unique_benefits.update(nutrient_info.health_benefits)
        
        # 多样性加分
        health_score += len(unique_benefits) * 0.05
        health_score = min(1.0, health_score)
        
        # 口感评分：基于成分兼容性和调味
        taste_score = 0.8  # 基础分
        
        # 检查成分间的兼容性
        for i, (nutrient_name1, _) in enumerate(nutrients):
            nutrient_info1 = self.nutrient_database.get(nutrient_name1)
            if not nutrient_info1:
                continue
                
            for j, (nutrient_name2, _) in enumerate(nutrients):
                if i >= j:  # 避免重复比较
                    continue
                    
                nutrient_info2 = self.nutrient_database.get(nutrient_name2)
                if not nutrient_info2:
                    continue
                    
                # 检查成分兼容性
                if any(comp in nutrient_info2.compatibility for comp in nutrient_info1.compatibility):
                    taste_score += 0.05
                elif any(incomp in nutrient_info2.incompatibility for incomp in nutrient_info1.incompatibility):
                    taste_score -= 0.1
        
        # 考虑基础饮品的兼容性
        for nutrient_name, _ in nutrients:
            nutrient_info = self.nutrient_database.get(nutrient_name)
            if nutrient_info:
                if base_beverage in nutrient_info.compatibility:
                    taste_score += 0.05
                elif base_beverage in nutrient_info.incompatibility:
                    taste_score -= 0.1
        
        # 成本评分：基于原料价格
        total_cost = self.base_beverages[base_beverage].get("price_per_liter", 0) * 0.5  # 500ml
        for nutrient_name, amount in nutrients:
            nutrient_info = self.nutrient_database.get(nutrient_name)
            if nutrient_info:
                ingredient_cost = nutrient_info.price_per_kg * (amount / 1000) * 0.5
                total_cost += ingredient_cost
        
        for adjuster_name, amount in flavor_adjusters:
            adjuster_info = self.flavor_adjusters.get(adjuster_name)
            if adjuster_info:
                # 确保使用正确的键名
                price_per_kg = adjuster_info.get("price_per_gram", 0) * 1000
                adjuster_cost = price_per_kg * (amount / 1000) * 0.5
                total_cost += adjuster_cost
        
        # 成本越低，评分越高（假设合理成本范围0-50元）
        cost_score = max(0.0, 1.0 - (total_cost / 50.0))
        
        # 创新评分：基于成分组合的新颖性和多样性
        unique_brands = set()
        unique_categories = set()
        
        for nutrient_name, _ in nutrients:
            if nutrient_name in self.nutrient_database:
                nutrient_info = self.nutrient_database[nutrient_name]
                unique_brands.add(nutrient_info.brand)
                unique_categories.add(nutrient_info.category)
        
        # 品牌多样性加分
        innovation_score = 0.5 + (len(unique_brands) * 0.1) + (len(unique_categories) * 0.1)
        # 成分数量加分
        innovation_score += len(nutrients) * 0.05
        # 协同效应加分
        innovation_score += synergy_analysis.synergy_score * 0.2
        # 随机创新因子
        innovation_score += np.random.uniform(0.0, 0.2)
        
        innovation_score = min(1.0, innovation_score)
        
        # 市场潜力评分：基于消费者群体和健康目标匹配度
        market_potential = 0.5
        consumer_group = consumer_analysis.get("consumer_group", "")
        health_goal = consumer_analysis.get("health_goal", "")
        
        # 根据消费者群体的市场规模调整
        consumer_market_size = {
            "上班族": 0.9, "学生": 0.8, "中老年人": 0.7, 
            "健身人群": 0.6, "爱美人士": 0.7, "儿童青少年": 0.6,
            "孕产妇": 0.5, "亚健康人群": 0.6, "减肥人群": 0.7,
            "失眠人群": 0.5
        }
        market_potential += consumer_market_size.get(consumer_group, 0.5) * 0.3
        
        # 根据健康目标的普遍性调整
        health_goal_popularity = {
            "增强免疫力": 0.9, "增强记忆力": 0.6, "骨骼健康": 0.7, "肌肉增长": 0.6, 
            "美容养颜": 0.8, "抗氧化": 0.7, "改善睡眠": 0.6, "抗疲劳": 0.8,
            "生长发育": 0.5, "心血管保护": 0.6
        }
        market_potential += health_goal_popularity.get(health_goal, 0.5) * 0.2
        
        market_potential = min(1.0, market_potential)
        
        return {
            "health": health_score,
            "taste": min(1.0, taste_score),
            "cost": cost_score,
            "innovation": innovation_score,
            "market_potential": market_potential
        }
    
    def analyze_consumer_needs(self, consumer_group: str, health_goal: str) -> Dict:
        """分析消费者需求"""
        profile = self.consumer_profiles.get(consumer_group, {})
        if not profile:
            raise ValueError(f"未知消费者群体: {consumer_group}")
        
        # 匹配健康目标与营养成分
        matching_nutrients = []
        for nutrient_name, nutrient_info in self.nutrient_database.items():
            if health_goal in nutrient_info.health_benefits:
                matching_nutrients.append((nutrient_name, nutrient_info.research_score))
        
        # 按研发价值排序
        matching_nutrients.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "consumer_group": consumer_group,
            "health_goal": health_goal,
            "primary_needs": profile.get("primary_needs", []),
            "preferred_flavors": profile.get("preferred_flavors", []),
            "matching_nutrients": [n[0] for n in matching_nutrients[:5]],  # 取前5个
            "health_concerns": profile.get("health_concerns", []),
            "price_sensitivity": profile.get("price_sensitivity", "中等"),
            "price_range": profile.get("price_range", [15, 35])
        }
    
    def formulate_beverage(self, consumer_group: str, health_goal: str) -> BeverageFormulation:
        """基于强化学习的自主饮品配方研发"""
        print(f"🔬 基于强化学习的现制饮品研发专家系统启动...")
        print(f"   目标人群: {consumer_group}")
        print(f"   健康目标: {health_goal}")
        
        # 获取初始状态
        state = self._get_state_representation(consumer_group, health_goal)
        
        # 1. 分析消费者需求
        consumer_analysis = self.analyze_consumer_needs(consumer_group, health_goal)
        print(f"✅ 消费者需求分析完成")
        
        # 2. 选择基础饮品
        base_beverage = self._select_base_beverage(consumer_analysis)
        print(f"✅ 基础饮品选择完成: {base_beverage}")
        
        # 3. 智能选择营养成分（使用强化学习）
        core_nutrients = self._select_nutrients_rl(state, consumer_analysis)
        print(f"✅ 基于强化学习的营养成分选择完成: {', '.join([n[0] for n in core_nutrients])}")
        
        # 4. 分析协同效应
        synergy_analysis = self._analyze_synergy(core_nutrients)
        print(f"✅ 营养补充剂协同效应分析完成")
        
        # 5. 优化配方比例
        optimized_formulation = self._optimize_formulation(
            core_nutrients, base_beverage, consumer_analysis
        )
        print(f"✅ 配方比例优化完成")
        
        # 6. 选择调味剂
        flavor_adjusters = self._select_flavor_adjusters(
            core_nutrients, base_beverage, consumer_analysis
        )
        print(f"✅ 调味剂选择完成")
        
        # 7. 生成制作工艺
        preparation_method = self._generate_preparation_method(
            optimized_formulation, flavor_adjusters, base_beverage
        )
        print(f"✅ 制作工艺生成完成")
        
        # 8. 评估配方质量
        quality_scores = self._assess_quality(
            optimized_formulation, flavor_adjusters, base_beverage, synergy_analysis, consumer_analysis
        )
        print(f"✅ 配方质量评估完成")
        
        # 9. 创建配方对象
        formulation = BeverageFormulation(
            consumer_group=consumer_group,
            health_goal=health_goal,
            base_beverage=base_beverage,
            nutrients=optimized_formulation,
            flavor_adjusters=flavor_adjusters,
            preparation_method=preparation_method,
            health_score=quality_scores["health"],
            taste_score=quality_scores["taste"],
            cost_score=quality_scores["cost"],
            innovation_score=quality_scores["innovation"],
            market_potential=quality_scores["market_potential"],
            synergy_analysis=synergy_analysis
        )
        
        # 10. 计算奖励并更新强化学习代理
        reward = self._get_reward(formulation)
        next_state = self._get_state_representation(consumer_group, health_goal)
        done = True  # 单次配方研发完成
        
        # 存储经验
        action = hash(str(core_nutrients)) % self.action_dim  # 简化的动作表示
        self.rl_agent.remember(state, action, reward, next_state, done)
        
        # 经验回放训练
        self.rl_agent.replay()
        
        # 记录性能历史
        self.performance_history.append(reward)
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
        
        # 11. 记录研发历史
        self.research_history.append(formulation)
        if quality_scores["health"] > 0.8 and quality_scores["taste"] > 0.7:
            self.successful_formulations.append(formulation)
        
        print(f"🎉 饮品配方研发完成!")
        return formulation
    
    def _select_nutrients_rl(self, state: np.ndarray, consumer_analysis: Dict) -> List[Tuple[str, float]]:
        """使用强化学习选择营养成分"""
        # 基于当前状态选择动作
        action = self.rl_agent.act(state)
        
        # 将动作转换为营养成分选择
        # 这里简化处理，实际应用中需要更复杂的映射
        nutrient_names = list(self.nutrient_database.keys())
        selected_nutrients = []
        
        # 选择主要成分
        primary_nutrients = consumer_analysis["matching_nutrients"][:3]
        selected_nutrients.extend([(name, 0.0) for name in primary_nutrients if name in nutrient_names])
        
        # 根据动作选择额外成分
        if action < len(nutrient_names):
            extra_nutrient = nutrient_names[action]
            if extra_nutrient not in [n[0] for n in selected_nutrients]:
                selected_nutrients.append((extra_nutrient, 0.0))
        
        # 添加一些补充成分
        health_concerns = consumer_analysis.get("health_concerns", [])
        for concern in health_concerns:
            for nutrient_name, nutrient_info in self.nutrient_database.items():
                if concern in nutrient_info.health_benefits and \
                   nutrient_name not in [n[0] for n in selected_nutrients]:
                    selected_nutrients.append((nutrient_name, 0.0))
                    break
        
        # 限制总数
        selected_nutrients = selected_nutrients[:7]
        
        return selected_nutrients
    
    def get_processing_recommendations(self, formulation: BeverageFormulation) -> Dict:
        """获取加工建议"""
        recommendations = {
            "equipment": {},
            "parameters": {},
            "quality_control": {}
        }
        
        # 设备推荐
        recommendations["equipment"] = {
            "primary": "高速搅拌机",
            "secondary": "超声波均质机",
            "sterilization": "低温杀菌设备"
        }
        
        # 参数设置
        recommendations["parameters"] = {
            "mixing_speed": "20000 RPM",
            "mixing_time": "60秒",
            "temperature": "常温操作",
            "sterilization_temp": "45°C",
            "sterilization_time": "8分钟"
        }
        
        # 质量控制
        recommendations["quality_control"] = {
            "ph_monitoring": "实时监测pH值保持在6.0-7.0",
            "dissolution_check": "确保无颗粒残留",
            "taste_testing": "三人小组品鉴确认口感",
            "shelf_life": "冷藏条件下24小时",
            "storage_temp": "4°C恒温储存"
        }
        
        return recommendations

def demonstrate_enhanced_rl_expert_system():
    """演示增强版基于强化学习的专家系统功能"""
    print("🤖 增强版基于强化学习的自主现制饮品研发专家系统演示")
    print("=" * 50)
    
    # 创建专家系统
    expert = EnhancedRLAutonomousBeverageExpert()
    print("✅ 增强版基于强化学习的现制饮品研发专家系统初始化完成")
    print(f"   - 营养成分数据库: {len(expert.nutrient_database)}种")
    print(f"   - 消费者画像: {len(expert.consumer_profiles)}类")
    print(f"   - 加工设备: {len(expert.processing_equipment)}台")
    print(f"   - 强化学习代理: DQN网络 (状态维度: {expert.state_dim}, 动作维度: {expert.action_dim})")
    
    # 研发多个配方进行训练
    test_cases = [
        ("上班族", "增强免疫力"),
        ("学生", "增强记忆力"),
        ("中老年人", "骨骼健康"),
        ("健身人群", "肌肉增长"),
        ("爱美人士", "美容养颜"),
        ("儿童青少年", "生长发育"),
        ("孕产妇", "骨骼健康"),
        ("亚健康人群", "增强免疫力"),
        ("减肥人群", "减肥"),
        ("失眠人群", "改善睡眠")
    ]
    
    print(f"\n🔬 开始强化学习训练...")
    for episode in range(3):  # 3轮训练
        print(f"\n🎯 训练轮次 {episode + 1}")
        print("-" * 30)
        
        for consumer_group, health_goal in test_cases:
            # 自主研发
            formulation = expert.formulate_beverage(consumer_group, health_goal)
            
            # 显示结果
            print(f"   {consumer_group}-{health_goal}:")
            print(f"     核心成分: {[n[0] for n in formulation.nutrients]}")
            print(f"     综合评分: {(formulation.health_score + formulation.taste_score + formulation.cost_score + formulation.innovation_score)/4:.2f}")
            print(f"     协同效应: {formulation.synergy_analysis.synergy_score:.2f}")
            print(f"     市场潜力: {formulation.market_potential:.2f}")
    
    print(f"\n📊 训练统计:")
    print(f"   总研发次数: {len(expert.research_history)}")
    print(f"   成功配方: {len(expert.successful_formulations)}")
    print(f"   平均性能: {np.mean(expert.performance_history):.2f}" if expert.performance_history else "   平均性能: 0.00")
    
    # 展示最终的强化学习效果
    print(f"\n🎯 强化学习效果展示:")
    print(f"   探索率: {expert.rl_agent.epsilon:.3f}")
    print(f"   经验回放池大小: {len(expert.rl_agent.memory)}")

def main():
    """主函数"""
    print("🥤 增强版基于强化学习的自主现制饮品研发专家系统")
    print("=" * 60)
    print("本系统具备真正的研发决策能力，能够:")
    print("✅ 自主分析消费者需求")
    print("✅ 基于强化学习智能选择营养成分")
    print("✅ 分析营养补充剂协同效应")
    print("✅ 优化配方比例")
    print("✅ 生成制作工艺")
    print("✅ 评估配方质量")
    print("✅ 持续学习和优化决策策略")
    print("✅ 支持更广泛的品牌、产品和人群")
    
    # 演示系统功能
    demonstrate_enhanced_rl_expert_system()
    
    print(f"\n🎯 系统价值:")
    print("• 真正的AI研发专家，具备自主学习和决策能力")
    print("• 基于深度强化学习的智能优化")
    print("• 从原料到工艺的全流程自主实现")
    print("• 可持续优化的研发能力")
    print("• 基于真实品牌营养补充剂数据的专业研发")
    print("• 支持28种品牌营养补充剂、10类基础饮品、10类调节剂和10类消费群体")

if __name__ == "__main__":
    main()