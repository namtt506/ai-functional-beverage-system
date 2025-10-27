"""
功能性现制饮品市场分析模块
基于真实品牌营养补充剂数据进行市场趋势分析和商业价值评估
"""

import random
import numpy as np
from typing import Dict, List, Tuple
from brand_nutritional_supplement_dataset import (
    BRAND_NUTRITIONAL_SUPPLEMENTS,
    TARGET_CONSUMERS
)

class MarketAnalyzer:
    """市场分析器"""
    
    def __init__(self):
        """初始化市场分析器"""
        self.brand_data = BRAND_NUTRITIONAL_SUPPLEMENTS
        self.consumer_data = TARGET_CONSUMERS
        self.market_trends = self._analyze_market_trends()
        
    def _analyze_market_trends(self) -> Dict:
        """分析市场趋势"""
        # 基于品牌市场份额和用户评分计算趋势
        category_demand = {}
        brand_performance = {}
        
        for supplement_name, supplement_info in self.brand_data.items():
            category = supplement_info["category"]
            brand = supplement_info["brand"]
            market_share = supplement_info["market_share"]
            user_rating = supplement_info["user_rating"]
            
            # 计算类别需求指数
            if category not in category_demand:
                category_demand[category] = 0
            category_demand[category] += market_share * user_rating
            
            # 计算品牌表现指数
            if brand not in brand_performance:
                brand_performance[brand] = {
                    "market_share": 0,
                    "avg_rating": 0,
                    "product_count": 0
                }
            brand_performance[brand]["market_share"] += market_share
            brand_performance[brand]["avg_rating"] += user_rating
            brand_performance[brand]["product_count"] += 1
        
        # 计算品牌平均评分
        for brand in brand_performance:
            if brand_performance[brand]["product_count"] > 0:
                brand_performance[brand]["avg_rating"] /= brand_performance[brand]["product_count"]
        
        return {
            "category_demand": category_demand,
            "brand_performance": brand_performance
        }
    
    def get_category_insights(self) -> Dict:
        """获取类别洞察"""
        category_demand = self.market_trends["category_demand"]
        
        # 按需求指数排序
        sorted_categories = sorted(category_demand.items(), key=lambda x: x[1], reverse=True)
        
        insights = {
            "top_categories": sorted_categories[:3],
            "emerging_categories": [],  # 基于增长率的新兴类别
            "declining_categories": []  # 基于下降趋势的衰退类别
        }
        
        # 简单模拟新兴和衰退类别
        all_categories = list(category_demand.keys())
        if len(all_categories) >= 3:
            insights["emerging_categories"] = [all_categories[-1]]  # 假设最后一个为新兴
            insights["declining_categories"] = [all_categories[1]]  # 假设第二个为衰退
        
        return insights
    
    def get_brand_competitiveness(self) -> Dict:
        """获取品牌竞争力分析"""
        brand_performance = self.market_trends["brand_performance"]
        
        # 计算综合竞争力指数
        competitiveness_scores = {}
        for brand, metrics in brand_performance.items():
            # 综合竞争力 = 市场份额 * 平均评分 * 产品数量权重
            competitiveness = (metrics["market_share"] * metrics["avg_rating"] * 
                             np.log(metrics["product_count"] + 1))
            competitiveness_scores[brand] = competitiveness
        
        # 按竞争力排序
        sorted_brands = sorted(competitiveness_scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "top_brands": sorted_brands[:5],
            "market_concentration": sum([score for _, score in sorted_brands[:3]]) / sum(competitiveness_scores.values())
        }
    
    def estimate_market_potential(self, formulation: Dict) -> Dict:
        """估算配方市场潜力"""
        # 提取配方中的核心成分品牌
        formulation_brands = set()
        for nutrient_name, _ in formulation.get("nutrients", []):
            if nutrient_name in self.brand_data:
                formulation_brands.add(self.brand_data[nutrient_name]["brand"])
        
        # 计算市场覆盖率
        total_market_share = 0
        avg_brand_rating = 0
        brand_count = 0
        
        for brand in formulation_brands:
            if brand in self.market_trends["brand_performance"]:
                total_market_share += self.market_trends["brand_performance"][brand]["market_share"]
                avg_brand_rating += self.market_trends["brand_performance"][brand]["avg_rating"]
                brand_count += 1
        
        if brand_count > 0:
            avg_brand_rating /= brand_count
        
        # 市场潜力指数计算
        market_potential = total_market_share * avg_brand_rating * 10  # 调整系数
        
        # 目标人群市场规模估算
        target_group = formulation.get("consumer_group", "上班族")
        group_size_factor = {
            "上班族": 1.0,
            "学生": 0.8,
            "中老年人": 0.7,
            "健身人群": 0.6,
            "爱美人士": 0.75
        }.get(target_group, 0.5)
        
        adjusted_potential = market_potential * group_size_factor
        
        return {
            "market_potential_index": round(adjusted_potential, 2),
            "coverage_brands": list(formulation_brands),
            "target_group": target_group,
            "group_size_factor": group_size_factor
        }
    
    def get_pricing_strategy(self, formulation: Dict) -> Dict:
        """获取定价策略建议"""
        # 基于成本和市场定位计算建议价格
        base_cost = 0
        target_group = formulation.get("consumer_group", "上班族")
        
        # 计算基础成本
        base_beverage = formulation.get("base_beverage", "纯净水")
        from brand_nutritional_supplement_dataset import BASE_BEVERAGES, FLAVOR_ADJUSTERS
        if base_beverage in BASE_BEVERAGES:
            base_cost += BASE_BEVERAGES[base_beverage].get("price_per_liter", 0) * 0.5  # 500ml
        
        # 计算营养成分成本
        for nutrient_name, amount in formulation.get("nutrients", []):
            if nutrient_name in self.brand_data:
                price_per_gram = self.brand_data[nutrient_name]["price_per_gram"]
                base_cost += price_per_gram * amount
        
        # 计算调味剂成本
        for adjuster_name, amount in formulation.get("flavor_adjusters", []):
            if adjuster_name in FLAVOR_ADJUSTERS:
                price_per_gram = FLAVOR_ADJUSTERS[adjuster_name].get("price_per_gram", 0)
                base_cost += price_per_gram * amount
        
        # 基于目标人群的价格敏感度调整
        price_sensitivity = self.consumer_data.get(target_group, {}).get("price_sensitivity", "中等")
        sensitivity_multiplier = {
            "较高": 0.8,   # 价格敏感，定价较低
            "中等": 1.0,   # 正常定价
            "较低": 1.3    # 价格不敏感，可定价较高
        }.get(price_sensitivity, 1.0)
        
        # 计算建议零售价 (成本 * 3 * 敏感度系数)
        suggested_price = base_cost * 3 * sensitivity_multiplier
        
        # 确保价格在合理范围内
        min_price = self.consumer_data.get(target_group, {}).get("price_range", [10, 30])[0]
        max_price = self.consumer_data.get(target_group, {}).get("price_range", [10, 30])[1]
        
        suggested_price = max(min_price, min(max_price, suggested_price))
        
        return {
            "cost": round(base_cost, 2),
            "suggested_price": round(suggested_price, 2),
            "price_sensitivity": price_sensitivity,
            "profit_margin": round((suggested_price - base_cost) / suggested_price * 100, 1) if suggested_price > 0 else 0
        }
    
    def get_competitive_advantage(self, formulation: Dict) -> Dict:
        """获取竞争优势分析"""
        # 基于配方独特性和品牌组合分析竞争优势
        nutrients = formulation.get("nutrients", [])
        unique_ingredients = len(set([name for name, _ in nutrients]))
        
        # 品牌多样性指数
        brands = set()
        for nutrient_name, _ in nutrients:
            if nutrient_name in self.brand_data:
                brands.add(self.brand_data[nutrient_name]["brand"])
        brand_diversity = len(brands)
        
        # 健康功效覆盖度
        health_benefits = set()
        for nutrient_name, _ in nutrients:
            if nutrient_name in self.brand_data:
                health_benefits.update(self.brand_data[nutrient_name]["health_benefits"])
        benefit_coverage = len(health_benefits)
        
        # 综合优势指数
        advantage_index = (unique_ingredients * 0.3 + brand_diversity * 0.4 + benefit_coverage * 0.3) * 10
        
        return {
            "unique_ingredients": unique_ingredients,
            "brand_diversity": brand_diversity,
            "benefit_coverage": benefit_coverage,
            "advantage_index": round(advantage_index, 2),
            "strengths": self._identify_strengths(unique_ingredients, brand_diversity, benefit_coverage)
        }
    
    def _identify_strengths(self, unique_ingredients: int, brand_diversity: int, benefit_coverage: int) -> List[str]:
        """识别配方优势"""
        strengths = []
        
        if unique_ingredients >= 2:
            strengths.append("成分组合独特")
        if brand_diversity >= 2:
            strengths.append("品牌协同效应")
        if benefit_coverage >= 3:
            strengths.append("多重健康功效")
        if unique_ingredients >= 3 and brand_diversity >= 2:
            strengths.append("高创新性")
            
        return strengths if strengths else ["基础配方"]

def demonstrate_market_analysis():
    """演示市场分析功能"""
    print("📈 功能性现制饮品市场分析演示")
    print("=" * 50)
    
    # 创建市场分析器
    analyzer = MarketAnalyzer()
    print("✅ 市场分析器初始化完成")
    print(f"   - 品牌数据: {len(analyzer.brand_data)}个")
    print(f"   - 消费者群体: {len(analyzer.consumer_data)}类")
    
    # 类别洞察
    print(f"\n📊 类别洞察:")
    category_insights = analyzer.get_category_insights()
    print(f"   热门类别: {category_insights['top_categories']}")
    print(f"   新兴类别: {category_insights['emerging_categories']}")
    print(f"   衰退类别: {category_insights['declining_categories']}")
    
    # 品牌竞争力
    print(f"\n🏆 品牌竞争力:")
    brand_competitiveness = analyzer.get_brand_competitiveness()
    print(f"   领先品牌: {brand_competitiveness['top_brands']}")
    print(f"   市场集中度: {brand_competitiveness['market_concentration']:.2%}")
    
    # 模拟配方进行市场潜力分析
    sample_formulation = {
        "consumer_group": "上班族",
        "base_beverage": "椰子水",
        "nutrients": [
            ("汤臣倍健维生素C片", 0.2),
            ("善存多维元素片", 0.3)
        ],
        "flavor_adjusters": [
            ("柠檬汁", 2.0)
        ]
    }
    
    print(f"\n🔮 市场潜力评估:")
    market_potential = analyzer.estimate_market_potential(sample_formulation)
    print(f"   市场潜力指数: {market_potential['market_potential_index']}")
    print(f"   覆盖品牌: {market_potential['coverage_brands']}")
    print(f"   目标人群: {market_potential['target_group']}")
    
    print(f"\n💰 定价策略建议:")
    pricing_strategy = analyzer.get_pricing_strategy(sample_formulation)
    print(f"   成本: ¥{pricing_strategy['cost']}")
    print(f"   建议售价: ¥{pricing_strategy['suggested_price']}")
    print(f"   价格敏感度: {pricing_strategy['price_sensitivity']}")
    print(f"   利润率: {pricing_strategy['profit_margin']}%")
    
    print(f"\n⚡ 竞争优势分析:")
    competitive_advantage = analyzer.get_competitive_advantage(sample_formulation)
    print(f"   独特成分数: {competitive_advantage['unique_ingredients']}")
    print(f"   品牌多样性: {competitive_advantage['brand_diversity']}")
    print(f"   功效覆盖度: {competitive_advantage['benefit_coverage']}")
    print(f"   优势指数: {competitive_advantage['advantage_index']}")
    print(f"   核心优势: {competitive_advantage['strengths']}")

if __name__ == "__main__":
    demonstrate_market_analysis()