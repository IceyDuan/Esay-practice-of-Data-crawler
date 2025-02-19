import openai
import pandas as pd
openai.api_key = "input your key"

# 方法2：调用GPT帮助标签化
def label_comment_by_GPT(comment):
    prompt = f"""请根据以下评论内容对"申请难度"、"课程评价"和"态度倾向"进行自动标签化。
评论内容: {comment}
请返回格式为 JSON，例如:
{{"申请难度": "hard", "课程评价": "positive", "态度倾向": "negative"}}
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "你好，现在你是一个擅长文本分类的专家，我需要你对Reddit上用户的评论进行标签分类。"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2
    )
    return response.choices[0].message['content']

# 测试
if __name__ == "__main__":
    sample_comment = "我觉得这个项目申请起来竞争激烈，而且课程非常充实，但整体体验让我有点失望。"

    # 关键词和情感分析
    labels_keyword = label_comment_by_GPT(sample_comment)
    print("大模型标签：", labels_keyword)

# 对清洗处理好的数据进行标签化
df = pd.read_csv("清洗及处理好的数据.csv")

# 对每行的关键内容生成标签
def label_row(row):
    content = row.get("关键内容", "")
    if not isinstance(content, str) or len(content.strip()) == 0:
        return pd.Series({"申请难度": None, "课程评价": None, "态度倾向": None})
    labels = label_comment_by_GPT(content)
    return pd.Series(labels)

df[['申请难度', '课程评价', '态度倾向']] = df.apply(label_row, axis=1)

# 导出文件
df.to_csv("标签化后的最终数据.csv", index=False, encoding="utf-8-sig")
print("标签化完成")