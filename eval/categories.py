# mmlu
mmlu_name_en2zh = {}

mmlu_subcategories = {
    "abstract_algebra": ["math"],
    "anatomy": ["health"],
    "astronomy": ["physics"],
    "business_ethics": ["business"],
    "clinical_knowledge": ["health"],
    "college_biology": ["biology"],
    "college_chemistry": ["chemistry"],
    "college_computer_science": ["computer science"],
    "college_mathematics": ["math"],
    "college_medicine": ["health"],
    "college_physics": ["physics"],
    "computer_security": ["computer science"],
    "conceptual_physics": ["physics"],
    "econometrics": ["economics"],
    "electrical_engineering": ["engineering"],
    "elementary_mathematics": ["math"],
    "formal_logic": ["philosophy"],
    "global_facts": ["other"],
    "high_school_biology": ["biology"],
    "high_school_chemistry": ["chemistry"],
    "high_school_computer_science": ["computer science"],
    "high_school_european_history": ["history"],
    "high_school_geography": ["geography"],
    "high_school_government_and_politics": ["politics"],
    "high_school_macroeconomics": ["economics"],
    "high_school_mathematics": ["math"],
    "high_school_microeconomics": ["economics"],
    "high_school_physics": ["physics"],
    "high_school_psychology": ["psychology"],
    "high_school_statistics": ["math"],
    "high_school_us_history": ["history"],
    "high_school_world_history": ["history"],
    "human_aging": ["health"],
    "human_sexuality": ["culture"],
    "international_law": ["law"],
    "jurisprudence": ["law"],
    "logical_fallacies": ["philosophy"],
    "machine_learning": ["computer science"],
    "management": ["business"],
    "marketing": ["business"],
    "medical_genetics": ["health"],
    "miscellaneous": ["other"],
    "moral_disputes": ["philosophy"],
    "moral_scenarios": ["philosophy"],
    "nutrition": ["health"],
    "philosophy": ["philosophy"],
    "prehistory": ["history"],
    "professional_accounting": ["other"],
    "professional_law": ["law"],
    "professional_medicine": ["health"],
    "professional_psychology": ["psychology"],
    "public_relations": ["politics"],
    "security_studies": ["politics"],
    "sociology": ["culture"],
    "us_foreign_policy": ["politics"],
    "virology": ["health"],
    "world_religions": ["philosophy"],
}

mmlu_categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering"],
    "humanities": ["history", "philosophy", "law"],
    "social sciences": ["politics", "culture", "economics", "geography", "psychology"],
    "other (business, health, misc.)": ["other", "business", "health"],
}


# cmmlu
cmmlu_name_en2zh = {
    "agronomy": "农学",
    "anatomy": "解剖学",
    "ancient_chinese": "古汉语",
    "arts": "艺术学",
    "astronomy": "天文学",
    "business_ethics": "商业伦理",
    "chinese_civil_service_exam": "中国公务员考试",
    "chinese_driving_rule": "中国驾驶规则",
    "chinese_food_culture": "中国饮食文化",
    "chinese_foreign_policy": "中国外交政策",
    "chinese_history":"中国历史",
    "chinese_literature": "中国文学",
    "chinese_teacher_qualification": "中国教师资格",
    "clinical_knowledge": "临床知识",
    "college_actuarial_science":"大学精算学",
    "college_education":"大学教育学",
    "college_engineering_hydrology": "大学工程水文学",
    "college_law": "大学法律",
    "college_mathematics": "大学数学",
    "college_medical_statistics":"大学医学统计",
    "college_medicine": "大学医学",
    "computer_science": "计算机科学",
    "computer_security": "计算机安全",
    "conceptual_physics": "概念物理学",
    "construction_project_management": "建设工程管理",
    "economics": "经济学",
    "education": "教育学",
    "electrical_engineering": "电气工程",
    "elementary_chinese":"小学语文",
    "elementary_commonsense":"小学常识",
    "elementary_information_and_technology": "小学信息技术",
    "elementary_mathematics": "初等数学",
    "ethnology": "民族学",
    "food_science": "食品科学",
    "genetics": "遗传学",
    "global_facts": "全球事实",
    "high_school_biology": "高中生物",
    "high_school_chemistry": "高中化学",
    "high_school_geography": "高中地理",
    "high_school_mathematics": "高中数学",
    "high_school_physics": "高中物理学",
    "high_school_politics": "高中政治",
    "human_sexuality": "人类性行为",
    "international_law": "国际法学",
    "journalism": "新闻学",
    "jurisprudence": "法理学",
    "legal_and_moral_basis": "法律与道德基础",
    "logical": "逻辑学",
    "machine_learning": "机器学习",
    "management": "管理学",
    "marketing": "市场营销",
    "marxist_theory": "马克思主义理论",
    "modern_chinese": "现代汉语",
    "nutrition": "营养学",
    "philosophy": "哲学",
    "professional_accounting": "专业会计",
    "professional_law": "专业法学",
    "professional_medicine": "专业医学",
    "professional_psychology": "专业心理学",
    "public_relations": "公共关系",
    "security_study":"安全研究",
    "sociology": "社会学",
    "sports_science": "体育学",
    "traditional_chinese_medicine": "中医中药",
    "virology": "病毒学",
    "world_history":"世界历史",
    "world_religions": "世界宗教",
}

# 每个文件对应到的子类别
cmmlu_subcategories = {
    "agronomy": ['other'],
    "anatomy": ['biology'],
    "ancient_chinese": ['linguistics','china specific'],
    "arts": ['arts'],
    "astronomy": ['physics'],
    "business_ethics": ['business'],
    "chinese_civil_service_exam": ['politics','china specific'],
    "chinese_driving_rule": ['other','china specific'],
    "chinese_food_culture": ['culture','china specific'],
    "chinese_foreign_policy": ['politics','china specific'],
    "chinese_history":['history','china specific'],
    "chinese_literature": ['literature','china specific'],
    "chinese_teacher_qualification": ['education','china specific'],
    "college_actuarial_science":['math'],
    "college_education":['education'],
    "college_engineering_hydrology": ['engineering'],
    "college_law": ['law'],
    "college_mathematics": ['math'],
    "college_medical_statistics":['statistics'],
    "clinical_knowledge": ['other'],
    "college_medicine": ['other'],
    "computer_science": ['computer science'],
    "computer_security": ['other'],
    "conceptual_physics": ['physics'],
    "construction_project_management": ['other','china specific'],
    "economics": ['economics'],
    "education": ['education'],
    "elementary_chinese":['linguistics','china specific'],
    "elementary_commonsense":['other','china specific'],
    "elementary_information_and_technology": ['other'],
    "electrical_engineering": ['engineering'],
    "elementary_mathematics": ['math'],
    "ethnology": ['culture','china specific'],
    "food_science": ['other'],
    "genetics": ['biology'],
    "global_facts": ['global'],
    "high_school_biology": ['biology'],
    "high_school_chemistry": ['chemistry'],
    "high_school_geography": ['geography'],
    "high_school_mathematics": ['math'],
    "high_school_physics": ['physics'],
    "high_school_politics": ['politics','china specific'],
    "human_sexuality": ['other'],
    "international_law": ['law'],
    "journalism": ['sociology'],
    "jurisprudence": ['law'],
    "legal_and_moral_basis": ['other'],
    "logical": ['philosophy'],
    "machine_learning": ['computer science'],
    "management": ['business'],
    "marketing": ['business'],
    "marxist_theory": ['philosophy'],
    "modern_chinese": ['linguistics','china specific'],
    "nutrition": ['other'],
    "philosophy": ['philosophy'],
    "professional_accounting": ['business'],
    "professional_law": ['law'],
    "professional_medicine": ['other'],
    "professional_psychology": ['psychology'],
    "public_relations": ['politics'],
    "security_study": ['politics'],
    "sociology": ['culture'],
    "sports_science": ['other'],
    "traditional_chinese_medicine": ['other','china specific'],
    "virology": ['biology'],
    "world_history":['history'],
    "world_religions": ['global'],
}

cmmlu_categories = {
    "STEM": ["physics", "chemistry", "biology", "computer science", "math", "engineering", "statistics"],
    "Humanities": ["history", "philosophy", "law", "arts", "literature", "global"],
    "Social Science": ['linguistics',"business", "politics", "culture", "economics", "geography", "psychology", "education", "sociology"],
    "Other":["other"],
    "China specific": ["china specific"],
}

# ceval
ceval_name_en2zh = {
    "high_school_physics": "高中物理",
    "fire_engineer": "注册消防工程师",
    "computer_network": "计算机网络",
    "advanced_mathematics": "高等数学",
    "logic": "逻辑学",
    "middle_school_physics": "初中物理",
    "clinical_medicine": "临床医学",
    "probability_and_statistics": "概率统计",
    "ideological_and_moral_cultivation": "思想道德修养与法律基础",
    "operating_system": "操作系统",
    "middle_school_mathematics": "初中数学",
    "chinese_language_and_literature": "中国语言文学",
    "electrical_engineer": "注册电气工程师",
    "business_administration": "工商管理",
    "high_school_geography": "高中地理",
    "modern_chinese_history": "近代史纲要",
    "legal_professional": "法律职业资格",
    "middle_school_geography": "初中地理",
    "middle_school_chemistry": "初中化学",
    "high_school_biology": "高中生物",
    "high_school_chemistry": "高中化学",
    "physician": "医师资格",
    "high_school_chinese": "高中语文",
    "tax_accountant": "税务师",
    "high_school_history": "高中历史",
    "mao_zedong_thought": "毛泽东思想和中国特色社会主义理论概论",
    "high_school_mathematics": "高中数学",
    "professional_tour_guide": "导游资格",
    "veterinary_medicine": "兽医学",
    "environmental_impact_assessment_engineer": "环境影响评价工程师",
    "basic_medicine": "基础医学",
    "education_science": "教育学",
    "urban_and_rural_planner": "注册城乡规划师",
    "middle_school_biology": "初中生物",
    "plant_protection": "植物保护",
    "middle_school_history": "初中历史",
    "high_school_politics": "高中政治",
    "metrology_engineer": "注册计量师",
    "art_studies": "艺术学",
    "college_economics": "大学经济学",
    "college_chemistry": "大学化学",
    "law": "法学",
    "sports_science": "体育学",
    "civil_servant": "公务员",
    "college_programming": "大学编程",
    "middle_school_politics": "初中政治",
    "teacher_qualification": "教师资格",
    "computer_architecture": "计算机组成",
    "college_physics": "大学物理",
    "discrete_mathematics": "离散数学",
    "marxism": "马克思主义基本原理",
    "accountant": "注册会计师",
}

ceval_subcategories = {
    "computer_network": [
        "computer_network"
    ],
    "operating_system": [
        "operating_system"
    ],
    "computer_architecture": [
        "computer_architecture"
    ],
    "college_programming": [
        "college_programming"
    ],
    "college_physics": [
        "college_physics"
    ],
    "college_chemistry": [
        "college_chemistry"
    ],
    "advanced_mathematics": [
        "advanced_mathematics"
    ],
    "probability_and_statistics": [
        "probability_and_statistics"
    ],
    "discrete_mathematics": [
        "discrete_mathematics"
    ],
    "electrical_engineer": [
        "electrical_engineer"
    ],
    "metrology_engineer": [
        "metrology_engineer"
    ],
    "high_school_mathematics": [
        "high_school_mathematics"
    ],
    "high_school_physics": [
        "high_school_physics"
    ],
    "high_school_chemistry": [
        "high_school_chemistry"
    ],
    "high_school_biology": [
        "high_school_biology"
    ],
    "middle_school_mathematics": [
        "middle_school_mathematics"
    ],
    "middle_school_biology": [
        "middle_school_biology"
    ],
    "middle_school_physics": [
        "middle_school_physics"
    ],
    "middle_school_chemistry": [
        "middle_school_chemistry"
    ],
    "veterinary_medicine": [
        "veterinary_medicine"
    ],
    "college_economics": [
        "college_economics"
    ],
    "business_administration": [
        "business_administration"
    ],
    "marxism": [
        "marxism"
    ],
    "mao_zedong_thought": [
        "mao_zedong_thought"
    ],
    "education_science": [
        "education_science"
    ],
    "teacher_qualification": [
        "teacher_qualification"
    ],
    "high_school_politics": [
        "high_school_politics"
    ],
    "high_school_geography": [
        "high_school_geography"
    ],
    "middle_school_politics": [
        "middle_school_politics"
    ],
    "middle_school_geography": [
        "middle_school_geography"
    ],
    "modern_chinese_history": [
        "modern_chinese_history"
    ],
    "ideological_and_moral_cultivation": [
        "ideological_and_moral_cultivation"
    ],
    "logic": [
        "logic"
    ],
    "law": [
        "law"
    ],
    "chinese_language_and_literature": [
        "chinese_language_and_literature"
    ],
    "art_studies": [
        "art_studies"
    ],
    "professional_tour_guide": [
        "professional_tour_guide"
    ],
    "legal_professional": [
        "legal_professional"
    ],
    "high_school_chinese": [
        "high_school_chinese"
    ],
    "high_school_history": [
        "high_school_history"
    ],
    "middle_school_history": [
        "middle_school_history"
    ],
    "civil_servant": [
        "civil_servant"
    ],
    "sports_science": [
        "sports_science"
    ],
    "plant_protection": [
        "plant_protection"
    ],
    "basic_medicine": [
        "basic_medicine"
    ],
    "clinical_medicine": [
        "clinical_medicine"
    ],
    "urban_and_rural_planner": [
        "urban_and_rural_planner"
    ],
    "accountant": [
        "accountant"
    ],
    "fire_engineer": [
        "fire_engineer"
    ],
    "environmental_impact_assessment_engineer": [
        "environmental_impact_assessment_engineer"
    ],
    "tax_accountant": [
        "tax_accountant"
    ],
    "physician": [
        "physician"
    ]
}

ceval_categories = {
    "Other": [
        "civil_servant",
        "sports_science",
        "plant_protection",
        "basic_medicine",
        "clinical_medicine",
        "urban_and_rural_planner",
        "accountant",
        "fire_engineer",
        "environmental_impact_assessment_engineer",
        "tax_accountant",
        "physician"
    ],
    "STEM": [
        "computer_network",
        "operating_system",
        "computer_architecture",
        "college_programming",
        "college_physics",
        "college_chemistry",
        "advanced_mathematics",
        "probability_and_statistics",
        "discrete_mathematics",
        "electrical_engineer",
        "metrology_engineer",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_chemistry",
        "high_school_biology",
        "middle_school_mathematics",
        "middle_school_biology",
        "middle_school_physics",
        "middle_school_chemistry",
        "veterinary_medicine"
    ],
    "Social Science": [
        "college_economics",
        "business_administration",
        "marxism",
        "mao_zedong_thought",
        "education_science",
        "teacher_qualification",
        "high_school_politics",
        "high_school_geography",
        "middle_school_politics",
        "middle_school_geography"
    ],
    "Humanities": [
        "modern_chinese_history",
        "ideological_and_moral_cultivation",
        "logic",
        "law",
        "chinese_language_and_literature",
        "art_studies",
        "professional_tour_guide",
        "legal_professional",
        "high_school_chinese",
        "high_school_history",
        "middle_school_history"
    ]
}

