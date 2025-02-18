import re
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
# 难度判定词
APPLY_DIFFICULTY_KEYWORDS = {
    "hard": ["competitive", "rigorous", "strict", "low acceptance"],
    "easy": ["easy to apply", "low barrier", "high acceptance"]
}

# 课程评价判定词
COURSE_EVALUATION_KEYWORDS = {
    "positive": ["engaging", "informative", "practical", "high quality", "detailed"],
    "negative": ["boring", "dull", "difficult", "poor", "vague"]
}

# 态度倾向判定词
ATTITUDE_KEYWORDS = {
    "positive": ["recommend", "like", "enjoy", "appreciate"],
    "negative": ["disappoint", "bad", "terrible", "hate"]
}

# 关键词匹配
def keyword_label(text, keyword_dict):
    label_scores = {}
    for label, keywords in keyword_dict.items():
        count = 0
        for kw in keywords:
            if re.search(kw, text, re.IGNORECASE):
                count += 1
        label_scores[label] = count
    if max(label_scores.values()) > 0:
        return max(label_scores, key=label_scores.get)
    else:
        return None

# 情感分析（返回数值高则积极）
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']

    if compound >= 0.05:
        sentiment_label = "positive"
    elif compound <= -0.05:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"
    return compound, sentiment_label

# 标签化（包含三个标签）
def label_comment(comment):
    text = comment.strip()
    # 申请难度，默认返回 "medium"
    apply_label = keyword_label(text, APPLY_DIFFICULTY_KEYWORDS)
    difficulty = apply_label if apply_label is not None else "medium"

    # 课程评价（关键词匹配结果与情感分析一致，则采用关键词匹配结果，否则使用情感分析结果）
    course_label = keyword_label(text, COURSE_EVALUATION_KEYWORDS)
    _, course_sentiment = analyze_sentiment(text)
    course_evaluation = course_label if course_label == course_sentiment else course_sentiment

    # 态度倾向
    _, attitude_sentiment = analyze_sentiment(text)
    attitude = attitude_sentiment

    return {
        "申请难度": difficulty,
        "课程评价": course_evaluation,
        "态度倾向": attitude
    }

# 测试
if __name__ == "__main__":
    sample_comment = "我觉得这个项目申请起来竞争激烈，而且课程非常充实，但整体体验让我有点失望。"

    # 关键词和情感分析
    labels_keyword = label_comment(sample_comment)
    print("关键词+情感分析标签：", labels_keyword)

# 对清洗处理好的数据进行标签化
df = pd.read_csv("清洗及处理好的数据.csv")

# 对每行的关键内容生成标签
def label_row(row):
    content = row.get("关键内容", "")
    if not isinstance(content, str) or len(content.strip()) == 0:
        return pd.Series({"申请难度": None, "课程评价": None, "态度倾向": None})
    labels = label_comment(content)
    return pd.Series(labels)


df[['申请难度', '课程评价', '态度倾向']] = df.apply(label_row, axis=1)

# 导出文件
df.to_csv("标签化后的最终数据.csv", index=False, encoding="utf-8-sig")
print("标签化完成")