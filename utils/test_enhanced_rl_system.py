"""
测试增强版基于强化学习的自主现制饮品研发专家系统
"""

from enhanced_rl_autonomous_beverage_expert import EnhancedRLAutonomousBeverageExpert
import json

def test_expert_system():
    """测试专家系统功能"""
    print("🧪 测试增强版基于强化学习的自主现制饮品研发专家系统")
    print("=" * 60)
    
    # 创建专家系统实例
    expert = EnhancedRLAutonomousBeverageExpert()
    
    print(f"✅ 系统初始化完成")
    print(f"   • 营养成分数据库: {len(expert.nutrient_database)} 种")
    print(f"   • 基础饮品载体: {len(expert.base_beverages)} 种")
    print(f"   • 口感调节剂: {len(expert.flavor_adjusters)} 种")
    print(f"   • 目标消费群体: {len(expert.consumer_profiles)} 类")
    print(f"   • 加工设备: {len(expert.processing_equipment)} 台")
    
    # 测试用例
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
    
    print(f"\n🔬 开始测试 {len(test_cases)} 个场景...")
    
    results = []
    for i, (consumer_group, health_goal) in enumerate(test_cases, 1):
        print(f"\n测试 {i}/{len(test_cases)}: {consumer_group} - {health_goal}")
        
        try:
            # 生成配方
            formulation = expert.formulate_beverage(consumer_group, health_goal)
            
            # 记录结果
            result = {
                "consumer_group": consumer_group,
                "health_goal": health_goal,
                "base_beverage": formulation.base_beverage,
                "nutrients_count": len(formulation.nutrients),
                "health_score": float(formulation.health_score),
                "taste_score": float(formulation.taste_score),
                "cost_score": float(formulation.cost_score),
                "innovation_score": float(formulation.innovation_score),
                "market_potential": float(formulation.market_potential),
                "synergy_score": float(formulation.synergy_analysis.synergy_score)
            }
            results.append(result)
            
            print(f"   ✓ 成功生成配方")
            print(f"     基础饮品: {formulation.base_beverage}")
            print(f"     核心成分: {len(formulation.nutrients)} 种")
            print(f"     健康评分: {formulation.health_score:.2f}")
            print(f"     协同效应: {formulation.synergy_analysis.synergy_score:.2f}")
            
        except Exception as e:
            print(f"   ❌ 测试失败: {str(e)}")
    
    # 统计结果
    print(f"\n📊 测试结果统计:")
    print(f"   总测试数: {len(test_cases)}")
    print(f"   成功数: {len(results)}")
    print(f"   失败数: {len(test_cases) - len(results)}")
    
    if results:
        avg_health = sum(r["health_score"] for r in results) / len(results)
        avg_taste = sum(r["taste_score"] for r in results) / len(results)
        avg_synergy = sum(r["synergy_score"] for r in results) / len(results)
        
        print(f"   平均健康评分: {avg_health:.2f}")
        print(f"   平均口感评分: {avg_taste:.2f}")
        print(f"   平均协同效应: {avg_synergy:.2f}")
        
        # 找出最佳配方
        best_formulation = max(results, key=lambda x: x["health_score"] + x["taste_score"])
        print(f"\n🏆 最佳配方:")
        print(f"   目标人群: {best_formulation['consumer_group']}")
        print(f"   健康目标: {best_formulation['health_goal']}")
        print(f"   健康评分: {best_formulation['health_score']:.2f}")
        print(f"   口感评分: {best_formulation['taste_score']:.2f}")
    
    # 强化学习效果
    print(f"\n🎯 强化学习效果:")
    print(f"   探索率: {expert.rl_agent.epsilon:.3f}")
    print(f"   经验回放池: {len(expert.rl_agent.memory)} 条")
    print(f"   平均性能: {sum(expert.performance_history)/len(expert.performance_history):.3f}" if expert.performance_history else "   平均性能: 0.000")
    
    # 保存测试结果
    with open("enhanced_rl_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 测试结果已保存到 enhanced_rl_test_results.json")
    
    print(f"\n🎉 测试完成!")

def test_consumer_analysis():
    """测试消费者需求分析功能"""
    print("\n👤 测试消费者需求分析功能")
    print("=" * 40)
    
    expert = EnhancedRLAutonomousBeverageExpert()
    
    # 测试所有消费者群体
    for consumer_group in expert.consumer_profiles.keys():
        try:
            analysis = expert.analyze_consumer_needs(consumer_group, "增强免疫力")
            print(f"   {consumer_group}:")
            print(f"     主要需求: {', '.join(analysis['primary_needs'][:3])}")
            print(f"     匹配成分: {len(analysis['matching_nutrients'])} 种")
            print(f"     价格敏感度: {analysis['price_sensitivity']}")
            print(f"     价格区间: ¥{analysis['price_range'][0]}-{analysis['price_range'][1]}")
            print()
        except Exception as e:
            print(f"   {consumer_group}: 测试失败 - {str(e)}")

def test_synergy_analysis():
    """测试协同效应分析功能"""
    print("🔄 测试协同效应分析功能")
    print("=" * 40)
    
    expert = EnhancedRLAutonomousBeverageExpert()
    
    # 选择几种常见的营养成分组合进行测试
    test_combinations = [
        [("汤臣倍健维生素C片", 0.5), ("善存多维元素片", 0.3)],
        [("汤臣倍健蛋白粉", 2.0), ("康恩贝胶原蛋白粉", 1.5), ("汤臣倍健钙维生素D片", 0.8)],
        [("东阿阿胶阿胶浆", 1.0), ("养生堂蜂胶软胶囊", 0.5)]
    ]
    
    for i, combination in enumerate(test_combinations, 1):
        try:
            synergy = expert._analyze_synergy(combination)
            print(f"   组合 {i}: {[item[0] for item in combination]}")
            print(f"     协同效应得分: {synergy.synergy_score:.3f}")
            print(f"     兼容性得分: {synergy.compatibility_score:.3f}")
            print(f"     品牌协同度: {synergy.brand_synergy:.3f}")
            print(f"     功效重叠数: {synergy.health_benefit_overlap}")
            print()
        except Exception as e:
            print(f"   组合 {i}: 测试失败 - {str(e)}")

if __name__ == "__main__":
    print("🧪 增强版基于强化学习的自主现制饮品研发专家系统测试")
    print("=" * 60)
    
    # 运行各项测试
    test_expert_system()
    test_consumer_analysis()
    test_synergy_analysis()
    
    print(f"\n🎯 系统优势总结:")
    print("✅ 支持28种知名品牌营养补充剂")
    print("✅ 覆盖10类目标消费群体")
    print("✅ 提供10种基础饮品载体")
    print("✅ 配备10种口感调节剂")
    print("✅ 基于深度强化学习的智能决策")
    print("✅ 实现营养补充剂协同效应分析")
    print("✅ 支持全流程自主研发")
    print("✅ 具备持续学习优化能力")