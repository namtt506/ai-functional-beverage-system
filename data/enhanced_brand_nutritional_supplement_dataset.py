"""
增强版品牌营养补充剂数据集
包含更多品牌、更多产品和更多人群的数据，支持更广泛的饮品研发
"""

import random
import json
from typing import Dict, List, Tuple

# 增强版品牌营养补充剂数据集
ENHANCED_BRAND_NUTRITIONAL_SUPPLEMENTS = {
    # 汤臣倍健 (By-health) - 扩展产品线
    "汤臣倍健维生素C片": {
        "brand": "汤臣倍健",
        "category": "维生素",
        "main_ingredient": "维生素C",
        "content": "1000mg/片",
        "price_per_unit": 89.0,
        "units_per_package": 100,
        "price_per_gram": 0.89,
        "solubility": "水溶性",
        "flavor_profile": "酸味",
        "usage_rate": "0.1-0.3%",
        "health_benefits": ["增强免疫力", "抗氧化", "促进胶原蛋白合成"],
        "compatibility": ["柑橘类水果", "蜂蜜", "绿茶"],
        "incompatibility": ["牛奶", "钙质"],
        "processing_characteristics": {
            "dissolution_time": "30-60秒",
            "stability": "对热敏感",
            "ph_stability_range": "5.0-7.0"
        },
        "market_share": 0.15,
        "user_rating": 4.5
    },
    "汤臣倍健蛋白粉": {
        "brand": "汤臣倍健",
        "category": "蛋白质",
        "main_ingredient": "乳清蛋白",
        "content": "450g",
        "price_per_unit": 198.0,
        "units_per_package": 450,
        "price_per_gram": 0.44,
        "solubility": "水溶性",
        "flavor_profile": "奶香",
        "usage_rate": "2-8%",
        "health_benefits": ["肌肉增长", "体力恢复", "营养补充"],
        "compatibility": ["香蕉", "浆果", "巧克力"],
        "incompatibility": ["酸性过强的果汁"],
        "processing_characteristics": {
            "dissolution_time": "60-120秒",
            "stability": "稳定",
            "ph_stability_range": "6.0-8.0"
        },
        "market_share": 0.12,
        "user_rating": 4.3
    },
    "汤臣倍健鱼油软胶囊": {
        "brand": "汤臣倍健",
        "category": "植物提取物",
        "main_ingredient": "鱼油",
        "content": "1000mg/粒×100粒",
        "price_per_unit": 128.0,
        "units_per_package": 100,
        "price_per_gram": 1.28,
        "solubility": "脂溶性",
        "flavor_profile": "鱼腥味",
        "usage_rate": "0.5-2%",
        "health_benefits": ["心血管健康", "脑部发育", "抗炎"],
        "compatibility": ["柠檬汁", "蜂蜜", "椰奶"],
        "incompatibility": ["高温"],
        "processing_characteristics": {
            "dissolution_time": "需乳化处理",
            "stability": "对光敏感",
            "ph_stability_range": "6.5-7.5"
        },
        "market_share": 0.18,
        "user_rating": 4.6
    },
    "汤臣倍健钙维生素D片": {
        "brand": "汤臣倍健",
        "category": "矿物质",
        "main_ingredient": "钙+维生素D",
        "content": "600mg钙+200IU维生素D/片×120片",
        "price_per_unit": 158.0,
        "units_per_package": 120,
        "price_per_gram": 1.32,
        "solubility": "微溶",
        "flavor_profile": "无味",
        "usage_rate": "0.3-0.8%",
        "health_benefits": ["骨骼健康", "牙齿健康", "肌肉功能"],
        "compatibility": ["柠檬汁", "酸奶", "坚果类"],
        "incompatibility": ["草酸含量高的食物"],
        "processing_characteristics": {
            "dissolution_time": "120-180秒",
            "stability": "稳定",
            "ph_stability_range": "6.0-8.0"
        },
        "market_share": 0.14,
        "user_rating": 4.4
    },
    "汤臣倍健益生菌粉": {
        "brand": "汤臣倍健",
        "category": "益生菌",
        "main_ingredient": "益生菌复合粉",
        "content": "2g×30袋",
        "price_per_unit": 298.0,
        "units_per_package": 30,
        "price_per_gram": 4.97,
        "solubility": "水溶性",
        "flavor_profile": "微酸",
        "usage_rate": "0.2-0.5%",
        "health_benefits": ["肠道健康", "免疫调节", "营养吸收"],
        "compatibility": ["酸奶", "蜂蜜", "燕麦"],
        "incompatibility": ["高温", "抗生素"],
        "processing_characteristics": {
            "dissolution_time": "30-60秒",
            "stability": "对热敏感",
            "ph_stability_range": "6.5-7.5"
        },
        "market_share": 0.16,
        "user_rating": 4.5
    },
    
    # 善存 (Centrum) - 扩展产品线
    "善存多维元素片": {
        "brand": "善存",
        "category": "维生素",
        "main_ingredient": "复合维生素",
        "content": "60片",
        "price_per_unit": 99.0,
        "units_per_package": 60,
        "price_per_gram": 1.65,
        "solubility": "水溶性",
        "flavor_profile": "微苦",
        "usage_rate": "0.2-0.5%",
        "health_benefits": ["全面营养", "能量代谢", "免疫支持"],
        "compatibility": ["燕麦", "坚果", "蜂蜜"],
        "incompatibility": ["咖啡因"],
        "processing_characteristics": {
            "dissolution_time": "60-90秒",
            "stability": "稳定",
            "ph_stability_range": "6.0-8.0"
        },
        "market_share": 0.22,
        "user_rating": 4.4
    },
    "善存银善存片": {
        "brand": "善存",
        "category": "维生素",
        "main_ingredient": "专为中老年设计的维生素",
        "content": "60片",
        "price_per_unit": 109.0,
        "units_per_package": 60,
        "price_per_gram": 1.82,
        "solubility": "水溶性",
        "flavor_profile": "微苦",
        "usage_rate": "0.2-0.5%",
        "health_benefits": ["骨骼健康", "心血管保护", "认知功能"],
        "compatibility": ["花草茶", "坚果", "蜂蜜"],
        "incompatibility": ["铁质补充剂"],
        "processing_characteristics": {
            "dissolution_time": "60-90秒",
            "stability": "稳定",
            "ph_stability_range": "6.0-8.0"
        },
        "market_share": 0.19,
        "user_rating": 4.5
    },
    "善存男士多维片": {
        "brand": "善存",
        "category": "维生素",
        "main_ingredient": "专为男士设计的复合维生素",
        "content": "30片",
        "price_per_unit": 89.0,
        "units_per_package": 30,
        "price_per_gram": 2.97,
        "solubility": "水溶性",
        "flavor_profile": "微苦",
        "usage_rate": "0.2-0.5%",
        "health_benefits": ["体力恢复", "抗氧化", "前列腺健康"],
        "compatibility": ["咖啡", "坚果", "蜂蜜"],
        "incompatibility": ["酒精"],
        "processing_characteristics": {
            "dissolution_time": "60-90秒",
            "stability": "稳定",
            "ph_stability_range": "6.0-8.0"
        },
        "market_share": 0.08,
        "user_rating": 4.2
    },
    "善存女士多维片": {
        "brand": "善存",
        "category": "维生素",
        "main_ingredient": "专为女士设计的复合维生素",
        "content": "30片",
        "price_per_unit": 99.0,
        "units_per_package": 30,
        "price_per_gram": 3.30,
        "solubility": "水溶性",
        "flavor_profile": "微苦",
        "usage_rate": "0.2-0.5%",
        "health_benefits": ["美容养颜", "骨骼健康", "情绪调节"],
        "compatibility": ["红酒", "坚果", "蜂蜜"],
        "incompatibility": ["铁质补充剂"],
        "processing_characteristics": {
            "dissolution_time": "60-90秒",
            "stability": "稳定",
            "ph_stability_range": "6.0-8.0"
        },
        "market_share": 0.09,
        "user_rating": 4.3
    },
    
    # 康恩贝 (Conba) - 扩展产品线
    "康恩贝维生素E软胶囊": {
        "brand": "康恩贝",
        "category": "维生素",
        "main_ingredient": "维生素E",
        "content": "100mg/粒×100粒",
        "price_per_unit": 68.0,
        "units_per_package": 100,
        "price_per_gram": 0.68,
        "solubility": "脂溶性",
        "flavor_profile": "微甜",
        "usage_rate": "0.1-0.5%",
        "health_benefits": ["抗氧化", "美容养颜", "心血管保护"],
        "compatibility": ["坚果类", "植物奶", "蜂蜜"],
        "incompatibility": ["铁质补充剂"],
        "processing_characteristics": {
            "dissolution_time": "需乳化处理",
            "stability": "对光敏感",
            "ph_stability_range": "6.0-8.0"
        },
        "market_share": 0.08,
        "user_rating": 4.2
    },
    "康恩贝益生菌粉": {
        "brand": "康恩贝",
        "category": "益生菌",
        "main_ingredient": "益生菌复合粉",
        "content": "15袋",
        "price_per_unit": 158.0,
        "units_per_package": 15,
        "price_per_gram": 10.53,
        "solubility": "水溶性",
        "flavor_profile": "微酸",
        "usage_rate": "0.1-0.3%",
        "health_benefits": ["肠道健康", "免疫调节", "营养吸收"],
        "compatibility": ["酸奶", "蜂蜜", "燕麦"],
        "incompatibility": ["高温", "抗生素"],
        "processing_characteristics": {
            "dissolution_time": "30-60秒",
            "stability": "对热敏感",
            "ph_stability_range": "6.5-7.5"
        },
        "market_share": 0.11,
        "user_rating": 4.4
    },
    "康恩贝钙镁锌片": {
        "brand": "康恩贝",
        "category": "矿物质",
        "main_ingredient": "钙镁锌",
        "content": "100片",
        "price_per_unit": 78.0,
        "units_per_package": 100,
        "price_per_gram": 0.78,
        "solubility": "微溶",
        "flavor_profile": "无味",
        "usage_rate": "0.5-2%",
        "health_benefits": ["骨骼健康", "免疫支持", "生长发育"],
        "compatibility": ["柠檬汁", "酸奶", "坚果类"],
        "incompatibility": ["草酸含量高的食物"],
        "processing_characteristics": {
            "dissolution_time": "120-180秒",
            "stability": "稳定",
            "ph_stability_range": "6.0-8.0"
        },
        "market_share": 0.09,
        "user_rating": 4.1
    },
    "康恩贝胶原蛋白粉": {
        "brand": "康恩贝",
        "category": "蛋白质",
        "main_ingredient": "胶原蛋白肽",
        "content": "10g×30袋",
        "price_per_unit": 298.0,
        "units_per_package": 30,
        "price_per_gram": 0.99,
        "solubility": "水溶性",
        "flavor_profile": "微腥",
        "usage_rate": "1-3%",
        "health_benefits": ["皮肤健康", "关节保护", "头发强韧"],
        "compatibility": ["果汁", "蜂蜜", "椰奶"],
        "incompatibility": [],
        "processing_characteristics": {
            "dissolution_time": "60-120秒",
            "stability": "稳定",
            "ph_stability_range": "6.0-8.0"
        },
        "market_share": 0.13,
        "user_rating": 4.3
    },
    
    # 修正药业 - 扩展产品线
    "修正牌钙镁锌片": {
        "brand": "修正药业",
        "category": "矿物质",
        "main_ingredient": "钙镁锌",
        "content": "100片",
        "price_per_unit": 78.0,
        "units_per_package": 100,
        "price_per_gram": 0.78,
        "solubility": "微溶",
        "flavor_profile": "无味",
        "usage_rate": "0.5-2%",
        "health_benefits": ["骨骼健康", "免疫支持", "生长发育"],
        "compatibility": ["柠檬汁", "酸奶", "坚果类"],
        "incompatibility": ["草酸含量高的食物"],
        "processing_characteristics": {
            "dissolution_time": "120-180秒",
            "stability": "稳定",
            "ph_stability_range": "6.0-8.0"
        },
        "market_share": 0.09,
        "user_rating": 4.1
    },
    "修正牌维生素B族片": {
        "brand": "修正药业",
        "category": "维生素",
        "main_ingredient": "复合维生素B",
        "content": "60片",
        "price_per_unit": 58.0,
        "units_per_package": 60,
        "price_per_gram": 0.97,
        "solubility": "水溶性",
        "flavor_profile": "微苦",
        "usage_rate": "0.1-0.3%",
        "health_benefits": ["能量代谢", "神经系统健康", "抗压力"],
        "compatibility": ["坚果类", "香蕉", "燕麦"],
        "incompatibility": ["咖啡因"],
        "processing_characteristics": {
            "dissolution_time": "30-60秒",
            "stability": "稳定",
            "ph_stability_range": "6.0-8.0"
        },
        "market_share": 0.07,
        "user_rating": 4.0
    },
    "修正牌蛋白粉": {
        "brand": "修正药业",
        "category": "蛋白质",
        "main_ingredient": "大豆蛋白",
        "content": "500g",
        "price_per_unit": 168.0,
        "units_per_package": 500,
        "price_per_gram": 0.34,
        "solubility": "水溶性",
        "flavor_profile": "豆腥味",
        "usage_rate": "2-8%",
        "health_benefits": ["肌肉维护", "营养补充", "体力恢复"],
        "compatibility": ["香草", "可可", "坚果类"],
        "incompatibility": ["酸性过强的果汁"],
        "processing_characteristics": {
            "dissolution_time": "60-120秒",
            "stability": "稳定",
            "ph_stability_range": "6.0-8.0"
        },
        "market_share": 0.06,
        "user_rating": 3.9
    },
    
    # 东阿阿胶 - 扩展产品线
    "东阿阿胶阿胶浆": {
        "brand": "东阿阿胶",
        "category": "中药提取物",
        "main_ingredient": "阿胶",
        "content": "10支",
        "price_per_unit": 128.0,
        "units_per_package": 10,
        "price_per_gram": 12.8,
        "solubility": "水溶性",
        "flavor_profile": "甜腻",
        "usage_rate": "1-3%",
        "health_benefits": ["补血", "滋阴", "美容养颜"],
        "compatibility": ["红枣", "枸杞", "蜂蜜"],
        "incompatibility": ["萝卜", "茶"],
        "processing_characteristics": {
            "dissolution_time": "60-120秒",
            "stability": "稳定",
            "ph_stability_range": "6.0-8.0"
        },
        "market_share": 0.14,
        "user_rating": 4.3
    },
    "东阿阿胶阿胶糕": {
        "brand": "东阿阿胶",
        "category": "中药提取物",
        "main_ingredient": "阿胶糕",
        "content": "200g",
        "price_per_unit": 198.0,
        "units_per_package": 200,
        "price_per_gram": 0.99,
        "solubility": "水溶性",
        "flavor_profile": "甜腻",
        "usage_rate": "2-5%",
        "health_benefits": ["补血", "美容养颜", "调理月经"],
        "compatibility": ["红枣", "核桃", "蜂蜜"],
        "incompatibility": ["萝卜", "茶"],
        "processing_characteristics": {
            "dissolution_time": "需加热溶解",
            "stability": "稳定",
            "ph_stability_range": "6.0-8.0"
        },
        "market_share": 0.12,
        "user_rating": 4.2
    },
    
    # 无限极 - 扩展产品线
    "无限极增健口服液": {
        "brand": "无限极",
        "category": "中药提取物",
        "main_ingredient": "复合多糖",
        "content": "30支",
        "price_per_unit": 298.0,
        "units_per_package": 30,
        "price_per_gram": 9.93,
        "solubility": "水溶性",
        "flavor_profile": "甘甜",
        "usage_rate": "2-5%",
        "health_benefits": ["免疫调节", "抗疲劳", "改善睡眠"],
        "compatibility": ["蜂蜜", "柠檬汁", "花草茶"],
        "incompatibility": ["辛辣食物"],
        "processing_characteristics": {
            "dissolution_time": "即溶",
            "stability": "稳定",
            "ph_stability_range": "5.5-7.5"
        },
        "market_share": 0.07,
        "user_rating": 4.0
    },
    "无限极灵芝皇胶囊": {
        "brand": "无限极",
        "category": "中药提取物",
        "main_ingredient": "灵芝提取物",
        "content": "100粒",
        "price_per_unit": 398.0,
        "units_per_package": 100,
        "price_per_gram": 3.98,
        "solubility": "脂溶性",
        "flavor_profile": "微苦",
        "usage_rate": "0.5-2%",
        "health_benefits": ["抗疲劳", "改善睡眠", "护肝"],
        "compatibility": ["蜂蜜", "柠檬汁", "花草茶"],
        "incompatibility": ["辛辣食物"],
        "processing_characteristics": {
            "dissolution_time": "需乳化处理",
            "stability": "稳定",
            "ph_stability_range": "6.0-8.0"
        },
        "market_share": 0.05,
        "user_rating": 4.1
    },
    
    # 安利 (Amway) - 扩展产品线
    "安利纽崔莱维生素B族片": {
        "brand": "安利",
        "category": "维生素",
        "main_ingredient": "复合维生素B",
        "content": "90片",
        "price_per_unit": 168.0,
        "units_per_package": 90,
        "price_per_gram": 1.87,
        "solubility": "水溶性",
        "flavor_profile": "微苦",
        "usage_rate": "0.1-0.3%",
        "health_benefits": ["能量代谢", "神经系统健康", "抗压力"],
        "compatibility": ["坚果类", "香蕉", "燕麦"],
        "incompatibility": ["咖啡因"],
        "processing_characteristics": {
            "dissolution_time": "30-60秒",
            "stability": "稳定",
            "ph_stability_range": "6.0-8.0"
        },
        "market_share": 0.13,
        "user_rating": 4.5
    },
    "安利蛋白粉": {
        "brand": "安利",
        "category": "蛋白质",
        "main_ingredient": "大豆蛋白",
        "content": "500g",
        "price_per_unit": 298.0,
        "units_per_package": 500,
        "price_per_gram": 0.60,
        "solubility": "水溶性",
        "flavor_profile": "豆腥味",
        "usage_rate": "2-8%",
        "health_benefits": ["肌肉维护", "营养补充", "体力恢复"],
        "compatibility": ["香草", "可可", "坚果类"],
        "incompatibility": ["酸性过强的果汁"],
        "processing_characteristics": {
            "dissolution_time": "60-120秒",
            "stability": "稳定",
            "ph_stability_range": "6.0-8.0"
        },
        "market_share": 0.10,
        "user_rating": 4.4
    },
    "安利维生素C片": {
        "brand": "安利",
        "category": "维生素",
        "main_ingredient": "维生素C",
        "content": "120片",
        "price_per_unit": 198.0,
        "units_per_package": 120,
        "price_per_gram": 1.65,
        "solubility": "水溶性",
        "flavor_profile": "酸味",
        "usage_rate": "0.1-0.3%",
        "health_benefits": ["增强免疫力", "抗氧化", "促进胶原蛋白合成"],
        "compatibility": ["柑橘类水果", "蜂蜜", "绿茶"],
        "incompatibility": ["牛奶", "钙质"],
        "processing_characteristics": {
            "dissolution_time": "30-60秒",
            "stability": "对热敏感",
            "ph_stability_range": "5.0-7.0"
        },
        "market_share": 0.11,
        "user_rating": 4.3
    },
    "安利钙镁片": {
        "brand": "安利",
        "category": "矿物质",
        "main_ingredient": "钙镁",
        "content": "200片",
        "price_per_unit": 298.0,
        "units_per_package": 200,
        "price_per_gram": 1.49,
        "solubility": "微溶",
        "flavor_profile": "无味",
        "usage_rate": "0.5-2%",
        "health_benefits": ["骨骼健康", "肌肉功能", "神经系统"],
        "compatibility": ["柠檬汁", "酸奶", "坚果类"],
        "incompatibility": ["草酸含量高的食物"],
        "processing_characteristics": {
            "dissolution_time": "120-180秒",
            "stability": "稳定",
            "ph_stability_range": "6.0-8.0"
        },
        "market_share": 0.09,
        "user_rating": 4.2
    },
    
    # 新增品牌：养生堂
    "养生堂天然维生素E软胶囊": {
        "brand": "养生堂",
        "category": "维生素",
        "main_ingredient": "天然维生素E",
        "content": "100mg/粒×100粒",
        "price_per_unit": 88.0,
        "units_per_package": 100,
        "price_per_gram": 0.88,
        "solubility": "脂溶性",
        "flavor_profile": "微甜",
        "usage_rate": "0.1-0.5%",
        "health_benefits": ["抗氧化", "美容养颜", "心血管保护"],
        "compatibility": ["坚果类", "植物奶", "蜂蜜"],
        "incompatibility": ["铁质补充剂"],
        "processing_characteristics": {
            "dissolution_time": "需乳化处理",
            "stability": "对光敏感",
            "ph_stability_range": "6.0-8.0"
        },
        "market_share": 0.06,
        "user_rating": 4.1
    },
    "养生堂蜂胶软胶囊": {
        "brand": "养生堂",
        "category": "中药提取物",
        "main_ingredient": "蜂胶提取物",
        "content": "500mg/粒×100粒",
        "price_per_unit": 168.0,
        "units_per_package": 100,
        "price_per_gram": 1.68,
        "solubility": "脂溶性",
        "flavor_profile": "微苦",
        "usage_rate": "0.2-0.8%",
        "health_benefits": ["增强免疫力", "抗菌消炎", "口腔健康"],
        "compatibility": ["蜂蜜", "柠檬汁", "绿茶"],
        "incompatibility": ["高温"],
        "processing_characteristics": {
            "dissolution_time": "需乳化处理",
            "stability": "对光敏感",
            "ph_stability_range": "6.0-8.0"
        },
        "market_share": 0.05,
        "user_rating": 4.0
    },
    
    # 新增品牌：同仁堂
    "同仁堂六味地黄丸": {
        "brand": "同仁堂",
        "category": "中药提取物",
        "main_ingredient": "六味地黄丸提取物",
        "content": "9g×10丸",
        "price_per_unit": 88.0,
        "units_per_package": 10,
        "price_per_gram": 0.98,
        "solubility": "水溶性",
        "flavor_profile": "甘甜",
        "usage_rate": "1-3%",
        "health_benefits": ["滋阴补肾", "改善疲劳", "增强体质"],
        "compatibility": ["红枣", "枸杞", "蜂蜜"],
        "incompatibility": ["萝卜", "茶"],
        "processing_characteristics": {
            "dissolution_time": "需加热溶解",
            "stability": "稳定",
            "ph_stability_range": "6.0-8.0"
        },
        "market_share": 0.04,
        "user_rating": 4.2
    },
    "同仁堂阿胶": {
        "brand": "同仁堂",
        "category": "中药提取物",
        "main_ingredient": "阿胶",
        "content": "250g",
        "price_per_unit": 398.0,
        "units_per_package": 250,
        "price_per_gram": 1.59,
        "solubility": "水溶性",
        "flavor_profile": "甜腻",
        "usage_rate": "1-3%",
        "health_benefits": ["补血", "滋阴", "美容养颜"],
        "compatibility": ["红枣", "枸杞", "蜂蜜"],
        "incompatibility": ["萝卜", "茶"],
        "processing_characteristics": {
            "dissolution_time": "60-120秒",
            "stability": "稳定",
            "ph_stability_range": "6.0-8.0"
        },
        "market_share": 0.03,
        "user_rating": 4.3
    },
    
    # 新增品牌：碧生源
    "碧生源减肥茶": {
        "brand": "碧生源",
        "category": "中药提取物",
        "main_ingredient": "减肥茶提取物",
        "content": "2g×20袋",
        "price_per_unit": 68.0,
        "units_per_package": 20,
        "price_per_gram": 1.70,
        "solubility": "水溶性",
        "flavor_profile": "微苦",
        "usage_rate": "1-2%",
        "health_benefits": ["减肥", "排毒", "改善便秘"],
        "compatibility": ["柠檬汁", "蜂蜜", "花草茶"],
        "incompatibility": ["油腻食物"],
        "processing_characteristics": {
            "dissolution_time": "60-90秒",
            "stability": "稳定",
            "ph_stability_range": "6.0-8.0"
        },
        "market_share": 0.05,
        "user_rating": 3.8
    },
    
    # 新增品牌：脑白金
    "脑白金胶囊": {
        "brand": "脑白金",
        "category": "中药提取物",
        "main_ingredient": "褪黑素+淀粉",
        "content": "0.25g/粒×30粒",
        "price_per_unit": 198.0,
        "units_per_package": 30,
        "price_per_gram": 2.20,
        "solubility": "脂溶性",
        "flavor_profile": "无味",
        "usage_rate": "0.1-0.3%",
        "health_benefits": ["改善睡眠", "延缓衰老", "调节生物钟"],
        "compatibility": ["牛奶", "蜂蜜", "花草茶"],
        "incompatibility": ["咖啡因", "酒精"],
        "processing_characteristics": {
            "dissolution_time": "需乳化处理",
            "stability": "稳定",
            "ph_stability_range": "6.0-8.0"
        },
        "market_share": 0.04,
        "user_rating": 3.9
    }
}

# 增强版基础饮品载体数据
ENHANCED_BASE_BEVERAGES = {
    "纯净水": {
        "characteristics": "无色无味，适合所有营养成分",
        "price_per_liter": 0.5,
        "ph_value": 7.0,
        "mineral_content": "极低",
        "compatibility": "所有营养成分"
    },
    "椰子水": {
        "characteristics": "天然甜味，富含电解质",
        "price_per_liter": 8.0,
        "ph_value": 5.5,
        "mineral_content": "高",
        "compatibility": "维生素、矿物质、植物提取物"
    },
    "燕麦奶": {
        "characteristics": "奶香浓郁，质地顺滑",
        "price_per_liter": 12.0,
        "ph_value": 6.5,
        "mineral_content": "中等",
        "compatibility": "蛋白质、维生素E、植物提取物"
    },
    "杏仁奶": {
        "characteristics": "清淡香甜，低热量",
        "price_per_liter": 15.0,
        "ph_value": 6.0,
        "mineral_content": "低",
        "compatibility": "维生素、植物蛋白、植物提取物"
    },
    "绿茶": {
        "characteristics": "清香微苦，含天然抗氧化物",
        "price_per_liter": 6.0,
        "ph_value": 6.0,
        "mineral_content": "低",
        "compatibility": "蜂蜜、柠檬、姜黄素"
    },
    "花草茶": {
        "characteristics": "芳香怡人，多种口味选择",
        "price_per_liter": 7.0,
        "ph_value": 6.5,
        "mineral_content": "低",
        "compatibility": "蜂蜜、水果、维生素"
    },
    "豆浆": {
        "characteristics": "营养丰富，植物蛋白含量高",
        "price_per_liter": 10.0,
        "ph_value": 6.5,
        "mineral_content": "中等",
        "compatibility": "维生素、矿物质、植物提取物"
    },
    "牛奶": {
        "characteristics": "营养全面，钙质丰富",
        "price_per_liter": 15.0,
        "ph_value": 6.7,
        "mineral_content": "高",
        "compatibility": "维生素、矿物质、蛋白质"
    },
    "气泡水": {
        "characteristics": "清爽刺激，有助消化",
        "price_per_liter": 9.0,
        "ph_value": 4.5,
        "mineral_content": "低",
        "compatibility": "柠檬、薄荷、维生素C"
    },
    "果蔬汁": {
        "characteristics": "天然营养，口感丰富",
        "price_per_liter": 20.0,
        "ph_value": 5.0,
        "mineral_content": "高",
        "compatibility": "维生素、矿物质、植物提取物"
    }
}

# 增强版口感调节剂数据
ENHANCED_FLAVOR_ADJUSTERS = {
    "蜂蜜": {
        "function": "增甜、改善口感",
        "usage_rate": "2-8%",
        "price_per_gram": 0.045,
        "solubility": "水溶性",
        "flavor_profile": "甜味"
    },
    "柠檬汁": {
        "function": "增酸、提鲜、改善口感",
        "usage_rate": "1-5%",
        "price_per_gram": 0.012,
        "solubility": "水溶性",
        "flavor_profile": "酸味"
    },
    "香草精": {
        "function": "增香、掩盖不良味道",
        "usage_rate": "0.1-0.5%",
        "price_per_gram": 0.080,
        "solubility": "水溶性",
        "flavor_profile": "香草味"
    },
    "可可粉": {
        "function": "增加巧克力风味",
        "usage_rate": "1-5%",
        "price_per_gram": 0.065,
        "solubility": "水溶性",
        "flavor_profile": "巧克力味"
    },
    "薄荷提取物": {
        "function": "增加清凉感",
        "usage_rate": "0.05-0.2%",
        "price_per_gram": 0.120,
        "solubility": "水溶性",
        "flavor_profile": "薄荷味"
    },
    "姜汁": {
        "function": "增加辛辣感、促进血液循环",
        "usage_rate": "0.1-0.5%",
        "price_per_gram": 0.035,
        "solubility": "水溶性",
        "flavor_profile": "辛辣味"
    },
    "肉桂粉": {
        "function": "增加温暖感、调节血糖",
        "usage_rate": "0.05-0.2%",
        "price_per_gram": 0.055,
        "solubility": "水溶性",
        "flavor_profile": "辛香味"
    },
    "椰浆": {
        "function": "增加浓郁口感、提供植物脂肪",
        "usage_rate": "2-10%",
        "price_per_gram": 0.075,
        "solubility": "水溶性",
        "flavor_profile": "椰香味"
    },
    "枫糖浆": {
        "function": "天然增甜、增加独特风味",
        "usage_rate": "2-8%",
        "price_per_gram": 0.065,
        "solubility": "水溶性",
        "flavor_profile": "甜味"
    },
    "芦荟汁": {
        "function": "增加清爽感、护肤功效",
        "usage_rate": "1-5%",
        "price_per_gram": 0.085,
        "solubility": "水溶性",
        "flavor_profile": "清淡味"
    }
}

# 增强版目标消费群体数据
ENHANCED_TARGET_CONSUMERS = {
    "上班族": {
        "primary_needs": ["抗疲劳", "增强免疫力", "改善睡眠"],
        "preferred_brands": ["汤臣倍健", "善存", "安利"],
        "preferred_flavors": ["柑橘味", "浆果味", "香草味"],
        "price_sensitivity": "中等",
        "price_range": [15, 35],
        "consumption_habits": ["早晨饮用", "下午茶时间"],
        "health_concerns": ["免疫力低下", "睡眠质量差", "工作压力大"]
    },
    "学生": {
        "primary_needs": ["增强记忆力", "补充营养", "抗疲劳"],
        "preferred_brands": ["汤臣倍健", "安利", "善存"],
        "preferred_flavors": ["水果味", "巧克力味", "香草味"],
        "price_sensitivity": "较高",
        "price_range": [10, 25],
        "consumption_habits": ["学习时饮用", "运动后"],
        "health_concerns": ["记忆力不足", "营养不均衡", "学习压力大"]
    },
    "中老年人": {
        "primary_needs": ["骨骼健康", "心血管保护", "免疫调节"],
        "preferred_brands": ["善存", "汤臣倍健", "修正药业"],
        "preferred_flavors": ["温和口味", "花草茶味", "坚果味"],
        "price_sensitivity": "较低",
        "price_range": [20, 45],
        "consumption_habits": ["餐后饮用", "睡前饮用"],
        "health_concerns": ["骨质疏松", "心血管疾病", "免疫力下降"]
    },
    "健身人群": {
        "primary_needs": ["肌肉增长", "体力恢复", "蛋白质补充"],
        "preferred_brands": ["汤臣倍健", "安利", "康恩贝"],
        "preferred_flavors": ["香蕉味", "巧克力味", "浆果味"],
        "price_sensitivity": "较低",
        "price_range": [25, 50],
        "consumption_habits": ["运动后饮用", "训练前"],
        "health_concerns": ["肌肉不足", "体力下降", "恢复缓慢"]
    },
    "爱美人士": {
        "primary_needs": ["美容养颜", "抗氧化", "皮肤健康"],
        "preferred_brands": ["康恩贝", "汤臣倍健", "善存"],
        "preferred_flavors": ["花香", "水果味", "椰香"],
        "price_sensitivity": "较低",
        "price_range": [20, 40],
        "consumption_habits": ["美容护理时", "日常保养"],
        "health_concerns": ["皮肤老化", "色素沉淀", "肤质粗糙"]
    },
    "儿童青少年": {
        "primary_needs": ["生长发育", "智力发展", "营养补充"],
        "preferred_brands": ["汤臣倍健", "善存", "康恩贝"],
        "preferred_flavors": ["水果味", "巧克力味", "酸奶味"],
        "price_sensitivity": "较高",
        "price_range": [15, 30],
        "consumption_habits": ["早餐时间", "课间补充"],
        "health_concerns": ["营养不良", "发育迟缓", "注意力不集中"]
    },
    "孕产妇": {
        "primary_needs": ["营养补充", "胎儿发育", "产后恢复"],
        "preferred_brands": ["汤臣倍健", "善存", "东阿阿胶"],
        "preferred_flavors": ["温和口味", "水果味", "坚果味"],
        "price_sensitivity": "较低",
        "price_range": [30, 60],
        "consumption_habits": ["餐后饮用", "睡前饮用"],
        "health_concerns": ["营养不足", "贫血", "钙质缺乏"]
    },
    "亚健康人群": {
        "primary_needs": ["调理体质", "增强免疫力", "改善疲劳"],
        "preferred_brands": ["无限极", "同仁堂", "东阿阿胶"],
        "preferred_flavors": ["中药味", "温和口味", "花草茶味"],
        "price_sensitivity": "中等",
        "price_range": [25, 50],
        "consumption_habits": ["餐前饮用", "睡前饮用"],
        "health_concerns": ["体质虚弱", "免疫力低下", "慢性疲劳"]
    },
    "减肥人群": {
        "primary_needs": ["减肥", "排毒", "控制食欲"],
        "preferred_brands": ["碧生源", "康恩贝", "善存"],
        "preferred_flavors": ["清淡口味", "柠檬味", "薄荷味"],
        "price_sensitivity": "中等",
        "price_range": [20, 40],
        "consumption_habits": ["餐前饮用", "运动前"],
        "health_concerns": ["体重超标", "代谢缓慢", "便秘问题"]
    },
    "失眠人群": {
        "primary_needs": ["改善睡眠", "放松神经", "缓解压力"],
        "preferred_brands": ["脑白金", "无限极", "同仁堂"],
        "preferred_flavors": ["花草茶味", "温和口味", "牛奶味"],
        "price_sensitivity": "较低",
        "price_range": [30, 55],
        "consumption_habits": ["睡前饮用", "放松时刻"],
        "health_concerns": ["失眠", "焦虑", "神经紧张"]
    }
}

def get_brand_supplement_info(supplement_name: str) -> Dict:
    """获取特定品牌营养补充剂信息"""
    return ENHANCED_BRAND_NUTRITIONAL_SUPPLEMENTS.get(supplement_name, {})

def get_compatible_supplements(supplement_name: str) -> List[str]:
    """获取兼容的营养补充剂列表"""
    info = get_brand_supplement_info(supplement_name)
    if not info:
        return []
    
    compatibility_list = info.get("compatibility", [])
    compatible_supplements = []
    
    for name, supplement_info in ENHANCED_BRAND_NUTRITIONAL_SUPPLEMENTS.items():
        if name != supplement_name:
            # 检查成分兼容性
            if any(comp in supplement_info.get("compatibility", []) for comp in compatibility_list):
                compatible_supplements.append(name)
    
    return compatible_supplements

def get_incompatible_supplements(supplement_name: str) -> List[str]:
    """获取不兼容的营养补充剂列表"""
    info = get_brand_supplement_info(supplement_name)
    if not info:
        return []
    
    incompatibility_list = info.get("incompatibility", [])
    incompatible_supplements = []
    
    for name, supplement_info in ENHANCED_BRAND_NUTRITIONAL_SUPPLEMENTS.items():
        if name != supplement_name:
            # 检查成分不兼容性
            if any(incomp in supplement_info.get("incompatibility", []) for incomp in incompatibility_list):
                incompatible_supplements.append(name)
    
    return incompatible_supplements

def calculate_blend_cost(supplements: List[Tuple[str, float]]) -> float:
    """计算混合补充剂的成本"""
    total_cost = 0.0
    for supplement_name, amount in supplements:
        info = get_brand_supplement_info(supplement_name)
        if info:
            price_per_gram = info.get("price_per_gram", 0)
            total_cost += price_per_gram * amount
    
    return total_cost

def demonstrate_enhanced_dataset():
    """演示增强版数据集功能"""
    print("📊 增强版品牌营养补充剂数据集")
    print("=" * 60)
    
    print(f"✅ 数据集包含 {len(ENHANCED_BRAND_NUTRITIONAL_SUPPLEMENTS)} 种品牌营养补充剂:")
    brands = set()
    categories = set()
    for name, info in ENHANCED_BRAND_NUTRITIONAL_SUPPLEMENTS.items():
        brands.add(info['brand'])
        categories.add(info['category'])
        print(f"   • {name} ({info['brand']}) - {info['category']}")
    
    print(f"\n✅ 涵盖 {len(brands)} 个品牌: {', '.join(sorted(brands))}")
    print(f"✅ 涵盖 {len(categories)} 个类别: {', '.join(sorted(categories))}")
    
    print(f"\n✅ 基础饮品载体: {len(ENHANCED_BASE_BEVERAGES)} 种")
    for name, info in ENHANCED_BASE_BEVERAGES.items():
        print(f"   • {name}: {info['characteristics']}")
    
    print(f"\n✅ 口感调节剂: {len(ENHANCED_FLAVOR_ADJUSTERS)} 种")
    for name, info in ENHANCED_FLAVOR_ADJUSTERS.items():
        print(f"   • {name}: {info['function']}")
    
    print(f"\n✅ 目标消费群体: {len(ENHANCED_TARGET_CONSUMERS)} 类")
    for name, info in ENHANCED_TARGET_CONSUMERS.items():
        print(f"   • {name}: {', '.join(info['primary_needs'])}")
    
    # 演示兼容性分析
    print(f"\n🔍 兼容性分析示例:")
    test_supplement = "汤臣倍健维生素C片"
    compatible = get_compatible_supplements(test_supplement)
    incompatible = get_incompatible_supplements(test_supplement)
    
    print(f"   {test_supplement} 的兼容补充剂: {compatible[:5]}")
    print(f"   {test_supplement} 的不兼容补充剂: {incompatible[:3]}")
    
    # 演示成本计算
    print(f"\n💰 成本计算示例:")
    sample_blend = [
        ("汤臣倍健维生素C片", 0.5),  # 0.5克
        ("善存多维元素片", 0.3),     # 0.3克
        ("康恩贝益生菌粉", 0.2)      # 0.2克
    ]
    total_cost = calculate_blend_cost(sample_blend)
    print(f"   示例混合补充剂成本: ¥{total_cost:.2f}/克")

def save_enhanced_dataset_to_json():
    """将增强版数据集保存为JSON文件"""
    dataset = {
        "brand_nutritional_supplements": ENHANCED_BRAND_NUTRITIONAL_SUPPLEMENTS,
        "base_beverages": ENHANCED_BASE_BEVERAGES,
        "flavor_adjusters": ENHANCED_FLAVOR_ADJUSTERS,
        "target_consumers": ENHANCED_TARGET_CONSUMERS
    }
    
    with open("enhanced_brand_nutritional_supplement_dataset.json", "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)
    
    print("💾 增强版数据集已保存到 enhanced_brand_nutritional_supplement_dataset.json")

def main():
    """主函数"""
    print("🥤 增强版品牌营养补充剂数据集")
    print("=" * 60)
    print("本数据集包含:")
    print("• 28种知名品牌营养补充剂的详细特性")
    print("• 10种基础饮品载体")
    print("• 10种口感调节剂")
    print("• 10类目标消费群体画像")
    print("• 成分兼容性分析功能")
    print("• 成本计算功能")
    
    # 演示功能
    demonstrate_enhanced_dataset()
    
    # 保存数据集
    save_enhanced_dataset_to_json()
    
    print(f"\n🎯 应用价值:")
    print("• 为功能性现制饮品研发提供更丰富的数据支撑")
    print("• 支持更广泛的配方优化和创新")
    print("• 提供更精准的成分兼容性智能分析")
    print("• 实现更全面的成本控制和市场分析")

if __name__ == "__main__":
    main()