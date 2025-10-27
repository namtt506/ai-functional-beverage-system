import re
import json

# 读取数据集文件
with open('enhanced_brand_nutritional_supplement_dataset.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 提取所有健康目标
benefits = re.findall(r'"health_benefits": \[(.*?)\]', content)

# 收集所有唯一的健康目标
all_benefits = set()
for b in benefits:
    # 修复格式以便解析
    fixed_b = '[' + b + ']'
    try:
        benefit_list = json.loads(fixed_b)
        all_benefits.update(benefit_list)
    except:
        # 如果JSON解析失败，手动分割
        items = [item.strip().strip('"') for item in b.split(',')]
        all_benefits.update(items)

# 输出结果
print("健康目标列表:")
for benefit in sorted(all_benefits):
    print(f"  - {benefit}")

print(f"\n总计: {len(all_benefits)} 个健康目标")