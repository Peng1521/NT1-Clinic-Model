"""
基于Logistic回归的预测模型交互界面
支持问卷和简要两种输入模式
"""
import streamlit as st
import numpy as np
from model import LogisticPredictionModel

# 页面配置
st.set_page_config(
    page_title="1型发作性睡病临床特征预测模型",
    page_icon="📊",
    layout="wide"
)

# 初始化session state
if 'model' not in st.session_state:
    st.session_state.model = LogisticPredictionModel()
if 'input_mode' not in st.session_state:
    st.session_state.input_mode = "问卷采集"
if 'questionnaire_answers' not in st.session_state:
    st.session_state.questionnaire_answers = {}
if 'brief_answers' not in st.session_state:
    st.session_state.brief_answers = {}

# 标题
st.title("📊 1型发作性睡病临床特征预测模型")
st.markdown("---")

# 1. 定义切换回调函数
def on_mode_change():
    # 当 radio 变化时，这个函数会自动触发
    # st.session_state.input_mode 会根据 radio 的 key 自动更新
    pass

# 侧边栏：选择输入模式
st.sidebar.title("信息输入模式")
# 使用 key="input_mode" 直接与 session_state 绑定
st.sidebar.radio(
    " ",
    ["问卷采集", "简要信息"],
    key="input_mode"
)

# 显式定义变量，防止 NameError
input_mode = st.session_state.input_mode

def segmented_choice(label, options, key, default_idx=None):
    """
    更稳健的选择组件：增加空值处理和默认索引检查
    """
    # 确保 default 落在 options 范围内
    default_val = None
    if default_idx is not None and 0 <= default_idx < len(options):
        default_val = options[default_idx]
    
    try:
        # 尝试使用新版组件
        res = st.segmented_control(
            label,
            options=options,
            default=default_val,
            key=key
        )
    except Exception:
        # 回退到 radio
        res = st.radio(
            label,
            options=options,
            index=default_idx if default_idx is not None else 0,
            key=key,
            horizontal=True
        )
    return res

# 诊断阈值（固定）
threshold = 0.855
st.sidebar.markdown(f" ")

# 问卷模式
if input_mode == "问卷采集":
    st.header("📝 问卷采集模式")
    st.markdown("请根据您的情况填写以下问卷（均为单选必填项）：")

    missing = []

    # 性别（男=0，女=1）
    gender_options = ["男", "女"]
    gender_idx_default = st.session_state.questionnaire_answers.get("性别_idx", None)
    gender_choice = segmented_choice(
        "性别",
        options=gender_options,
        key="questionnaire_gender",
        default_idx=gender_idx_default
    )
    gender_idx = gender_options.index(gender_choice) if gender_choice is not None else None
    st.session_state.questionnaire_answers["性别_idx"] = gender_idx
    if gender_idx is None:
        missing.append("性别")
    else:
        st.session_state.questionnaire_answers["性别"] = 0 if gender_choice == "男" else 1

    # 年龄
    age_input = st.text_input(
        "年龄（岁）",
        value=st.session_state.questionnaire_answers.get("年龄_raw", ""),
        key="questionnaire_age"
    )
    st.session_state.questionnaire_answers["年龄_raw"] = age_input
    try:
        age_val = int(age_input)
        if age_val < 0 or age_val > 120:
            missing.append("年龄需在0-120之间")
    except ValueError:
        missing.append("年龄未填写或格式错误")
        age_val = None
    st.session_state.questionnaire_answers["年龄"] = age_val

    # 猝倒：三个子问题，≥2项视为"有"
    st.markdown("** **")
    cat_options = ["有", "无"]
    cat_idx1_default = st.session_state.questionnaire_answers.get("猝倒_q1_idx", None)
    cat_idx2_default = st.session_state.questionnaire_answers.get("猝倒_q2_idx", None)
    cat_idx3_default = st.session_state.questionnaire_answers.get("猝倒_q3_idx", None)
    cataplexy_q1 = segmented_choice(
        "当您大笑、生气或情绪激动时，您有没有经历过腿部肌无力或膝盖弯曲要跌倒的感觉？",
        options=cat_options,
        key="questionnaire_cataplexy_q1",
        default_idx=cat_idx1_default
    )
    cataplexy_q2 = segmented_choice(
        "当您大笑、生气或情绪激动时，您有没有经历过下巴松垂或下垂的感觉？",
        options=cat_options,
        key="questionnaire_cataplexy_q2",
        default_idx=cat_idx2_default
    )
    cataplexy_q3 = segmented_choice(
        "当您大笑、生气或情绪激动时，您有没有经历过头或肩膀突然无力、往下掉的感觉？",
        options=cat_options,
        key="questionnaire_cataplexy_q3",
        default_idx=cat_idx3_default
    )
    st.session_state.questionnaire_answers["猝倒_q1_idx"] = cat_options.index(cataplexy_q1) if cataplexy_q1 is not None else None
    st.session_state.questionnaire_answers["猝倒_q2_idx"] = cat_options.index(cataplexy_q2) if cataplexy_q2 is not None else None
    st.session_state.questionnaire_answers["猝倒_q3_idx"] = cat_options.index(cataplexy_q3) if cataplexy_q3 is not None else None
    cataplexy_flags = []
    for idx in [
        st.session_state.questionnaire_answers["猝倒_q1_idx"],
        st.session_state.questionnaire_answers["猝倒_q2_idx"],
        st.session_state.questionnaire_answers["猝倒_q3_idx"],
    ]:
        if idx is None:
            missing.append("猝倒相关问题未全部选择")
            cataplexy_flags.append(0)
        else:
            cataplexy_flags.append(1 if idx == 0 else 0)  # "有" 为索引0
    cataplexy_count = sum(cataplexy_flags)
    st.session_state.questionnaire_answers["猝倒"] = 1 if cataplexy_count >= 2 else 0

    # 通用二选项
    options_tri = ["有", "无"]

    def tri_choice(label, key_name):
        default_idx = st.session_state.questionnaire_answers.get(f"{key_name}_idx", None)
        choice = segmented_choice(
            label,
            options=options_tri,
            key=f"questionnaire_{key_name}",
            default_idx=default_idx
        )
        idx = options_tri.index(choice) if choice is not None else None
        st.session_state.questionnaire_answers[f"{key_name}_idx"] = idx
        if idx is None:
            missing.append(label)
        else:
            st.session_state.questionnaire_answers[key_name] = 0 if choice == "无" else 1

    # 按指定顺序：睡瘫、幻觉、打鼾、遗尿、性格变化、嗜睡家族史
    tri_choice("您近期有没有过觉得醒了但全身无法动弹的情况（俗称“鬼压床”）？", "睡瘫")
    tri_choice("您近期有没有过在快睡着或快醒来时出现幻觉（如听到、看到或接触到不存在的东西）？", "幻觉")
    tri_choice("您目前睡觉时是否有打呼噜？", "打鼾")
    tri_choice("您近期睡觉时有没有过尿床？", "遗尿")
    tri_choice("您近期（或患病后）是否有明显的性格改变？", "性格变化")
    tri_choice("您是否有亲属有明显白天犯困的情况？", "嗜睡家族史")
    
    # 计算按钮
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        calculate_btn = st.button("计算预测结果", type="primary", use_container_width=True)
    
    if calculate_btn:
        if missing:
            st.error("请完成所有题目。")
        else:
            feature_vector = [
                st.session_state.questionnaire_answers["猝倒"],
                st.session_state.questionnaire_answers["打鼾"],
                st.session_state.questionnaire_answers["性格变化"],
                st.session_state.questionnaire_answers["幻觉"],
                st.session_state.questionnaire_answers["遗尿"],
                st.session_state.questionnaire_answers["性别"],
                st.session_state.questionnaire_answers["嗜睡家族史"],
                st.session_state.questionnaire_answers["睡瘫"],
                st.session_state.questionnaire_answers["年龄"]
            ]
            
            result = st.session_state.model.calculate_from_inputs(feature_vector, threshold)
            
            st.markdown("---")
            st.subheader("📈 预测结果")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Logit值", f"{result['logit']:.4f}")
            
            with col2:
                st.metric("预测概率（P值）", f"{result['p_value']:.4f}")
            
            with col3:
                diagnosis_color = "🟢" if result['diagnosis'] == "诊断" else "🔴"
                st.metric("诊断结果", f"{diagnosis_color} {result['diagnosis']}")
            
            st.info(f"""
            **结果说明：**\n
            因1型发作性睡病为罕见疾病，本模型使用了目标特异度为99%下的诊断P值阈值（P=0.855)，以最大化控制假阳性结果；受此影响，约有20%~30%的1型发作性睡病个体会被漏诊（灵敏度为70%~80%），请结合其他临床信息综合判断。
            """)

# 简要模式
else:
    st.header("📋 简要信息模式")
    st.markdown("请根据您的情况填写以下问卷（均为单选必填项）")

    missing_brief = []

    # 性别（男=0，女=1）
    gender_options = ["男", "女"]
    gender_idx_default = st.session_state.brief_answers.get("性别_idx", None)
    gender_choice = segmented_choice(
        "性别",
        options=gender_options,
        key="brief_gender",
        default_idx=gender_idx_default
    )
    gender_idx = gender_options.index(gender_choice) if gender_choice is not None else None
    st.session_state.brief_answers["性别_idx"] = gender_idx
    if gender_idx is None:
        missing_brief.append("性别")
    else:
        st.session_state.brief_answers["性别"] = 0 if gender_choice == "男" else 1

    # 年龄
    age_input_b = st.text_input(
        "年龄（岁）",
        value=st.session_state.brief_answers.get("年龄_raw", ""),
        key="brief_age"
    )
    st.session_state.brief_answers["年龄_raw"] = age_input_b
    try:
        age_val_b = int(age_input_b)
        if age_val_b < 0 or age_val_b > 120:
            missing_brief.append("年龄需在0-120之间")
    except ValueError:
        missing_brief.append("年龄未填写或格式错误")
        age_val_b = None
    st.session_state.brief_answers["年龄"] = age_val_b

    # 猝倒：简要模式合并为单个问题
    st.markdown("** **")
    cat_options = ["有", "无"]
    cat_idx_default = st.session_state.brief_answers.get("猝倒_idx", None)
    cataplexy_brief = segmented_choice(
        "猝倒",
        options=cat_options,
        key="brief_cataplexy",
        default_idx=cat_idx_default
    )
    cat_idx = cat_options.index(cataplexy_brief) if cataplexy_brief is not None else None
    st.session_state.brief_answers["猝倒_idx"] = cat_idx
    if cat_idx is None:
        missing_brief.append("猝倒")
        st.session_state.brief_answers["猝倒"] = 0
    else:
        st.session_state.brief_answers["猝倒"] = 1 if cat_idx == 0 else 0  # "有" 为1,"无" 为0

    # 通用二选项
    options_tri = ["有", "无"]

    def tri_choice_brief(label, key_name):
        default_idx = st.session_state.brief_answers.get(f"{key_name}_idx", None)
        choice = segmented_choice(
            label,
            options=options_tri,
            key=f"brief_{key_name}",
            default_idx=default_idx
        )
        idx = options_tri.index(choice) if choice is not None else None
        st.session_state.brief_answers[f"{key_name}_idx"] = idx
        if idx is None:
            missing_brief.append(label)
        else:
            st.session_state.brief_answers[key_name] = 0 if choice == "无" else 1

    # 按指定顺序：睡瘫、幻觉、打鼾、遗尿、性格变化、嗜睡家族史
    tri_choice_brief("睡瘫", "睡瘫")
    tri_choice_brief("幻觉", "幻觉")
    tri_choice_brief("打鼾", "打鼾")
    tri_choice_brief("遗尿", "遗尿")
    tri_choice_brief("性格变化", "性格变化")
    tri_choice_brief("嗜睡家族史", "嗜睡家族史")

    # 计算按钮
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        calculate_btn = st.button("计算预测结果", type="primary", use_container_width=True)
    
    if calculate_btn:
        if missing_brief:
            st.error("请完成所有题目后再计算：\n" + "；".join(missing_brief))
        else:
            feature_vector = [
                st.session_state.brief_answers["猝倒"],
                st.session_state.brief_answers["打鼾"],
                st.session_state.brief_answers["性格变化"],
                st.session_state.brief_answers["幻觉"],
                st.session_state.brief_answers["遗尿"],
                st.session_state.brief_answers["性别"],
                st.session_state.brief_answers["嗜睡家族史"],
                st.session_state.brief_answers["睡瘫"],
                st.session_state.brief_answers["年龄"]
            ]
            
            result = st.session_state.model.calculate_from_inputs(feature_vector, threshold)
            
            st.markdown("---")
            st.subheader("📈 预测结果")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Logit值", f"{result['logit']:.4f}")
            
            with col2:
                st.metric("预测概率（P值）", f"{result['p_value']:.4f}")
            
            with col3:
                diagnosis_color = "🟢" if result['diagnosis'] == "诊断" else "🔴"
                st.metric("诊断结果", f"{diagnosis_color} {result['diagnosis']}")
            
            st.info(f"""
            **结果说明：**\n
            因1型发作性睡病为罕见疾病，本模型使用了目标特异度为99%下的诊断P值阈值（P=0.855)，以最大化控制假阳性结果；受此影响，约有20%~30%的1型发作性睡病个体会被漏诊（灵敏度为70%~80%），请结合其他临床信息综合判断。
            """)

# 页脚
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <small>1型发作性睡病临床特征预测模型 | 基于Logistic回归 | PMH 2025</small>
</div>
""", unsafe_allow_html=True)

