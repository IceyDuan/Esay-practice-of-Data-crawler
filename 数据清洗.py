import pandas as pd
import re
import ast

df = pd.read_csv("爬取的原始数据.csv", encoding="utf-8-sig")

# 去空格和特殊字符
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'(http[s]?://[^\s\'"]+)|(www\.[^\s\'"]+)', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 清洗
df['title'] = df['title'].apply(clean_text)
df['content'] = df['content'].apply(clean_text)
df['universities'] = df['universities'].apply(clean_text)

# 清洗评论
def clean_comments(comments_str):
    try:
        comments = ast.literal_eval(comments_str)
        if isinstance(comments, list):
            cleaned = [clean_text(comment) for comment in comments]
            return cleaned
    except Exception as e:
        return []
    return []

df['top_comments'] = df['top_comments'].apply(clean_comments)

# 处理大学名字
def list_to_str(item):
    if isinstance(item, list):
        return ", ".join(item)
    return item

df['universities'] = df['universities'].apply(list_to_str)

# 把标题、正文、评论合并成关键内容
def combine_content(row):
    comments_text = " ".join(row['top_comments']) if isinstance(row['top_comments'], list) else ""
    combined = f"{row['title']} {row['content']} {comments_text}"
    return clean_text(combined)

# 定义项目名称提取的函数
def extract_program(text):
    patterns = {
        "Master": r"\b(Master(?:'s)?|MSc|MS)\b",
        "PhD": r"\b(PhD|Doctorate)\b",
        "Bachelor": r"\b(Bachelor(?:'s)?|BS|BA)\b"
    }
    for prog, pattern in patterns.items():
        if re.search(pattern, text, re.IGNORECASE):
            return prog
    return None

df['关键内容'] = df.apply(combine_content, axis=1)
df['大学名称'] = df['universities']
df['专业名称'] = df['majors']
df['项目名称'] = df['关键内容'].apply(extract_program)

final_columns = ["大学名称", "项目名称", "专业名称", "关键内容"]
structured_df = df[final_columns].copy()

# 处理无效数据
structured_df = structured_df.dropna(subset=["关键内容"])
structured_df = structured_df[structured_df['关键内容'].apply(lambda x: len(x.split()) >= 3)]

# 处理噪音数据
# 降噪
def clean_noise(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[\s\.,;:!?]+', ' ', text)
    return text.strip()
# 判定（字母，数字，字符的比例）
def is_noisy(text, threshold=0.2):
    if not isinstance(text, str) or len(text) == 0:
        return True
    alnum_count = sum(c.isalnum() for c in text)
    ratio = alnum_count / len(text)
    return ratio < threshold

structured_df['关键内容'] = structured_df['关键内容'].apply(clean_noise)
structured_df = structured_df[~structured_df['关键内容'].apply(is_noisy)]

# 处理重复数据
structured_df.drop_duplicates(subset=["关键内容"], inplace=True)

#导出文件
structured_df.to_csv("清洗及处理好的数据.csv", index=False, encoding="utf-8-sig")
print("数据处理完成")