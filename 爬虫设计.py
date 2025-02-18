import praw
import pandas as pd
import spacy

# 调用Reddit API
REDDIT_CLIENT_ID = "input your ID"
REDDIT_CLIENT_SECRET = "input your SECRET"
REDDIT_USER_AGENT = "input user agent"

# 设置Subreddits，与大学或专业相关的用户讨论
TARGET_SUBREDDITS = ["ApplyingToCollege", "gradadmissions", "studyabroad"]

# 利用spacy动态提取肯能存在的大学名称
nlp = spacy.load("en_core_web_sm")

def extract_universities(text):
    doc = nlp(text)
    # 利用ORG识别文本中的机构，即大学名字
    candidate_unis = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    filtered_unis = [uni for uni in candidate_unis if any(keyword in uni for keyword in ["University", "College", "Institute"])]
    return list(set(filtered_unis))  # 去重列表

# 设定相关专业（这里把目标放在与本人相关的专业了）
MAJORS = ["Computer Science", "Data Science", "Engineering", "Business"]

def extract_majors(text):
    found_majors = [major for major in MAJORS if major.lower() in text.lower()]
    return ", ".join(found_majors) if found_majors else None

# 爬虫函数本体（爬取前面Subreddit的帖子，用spacy提取大学名称）
def scrape_reddit_posts(subreddit_name, limit=50):
    subreddit = reddit.subreddit(subreddit_name)
    posts_data = []

    for post in subreddit.hot(limit=limit):  # 热门帖
        title = post.title
        content = post.selftext
        combined_text = f"{title}\n{content}"  # 合并标题正文

        # spacy提取大学名称
        extracted_universities = extract_universities(combined_text)
        # 提取专业名称
        extracted_majors = extract_majors(combined_text)

        # 热度前5的评论（高热度评论参考价值大，避免数据冗杂）
        post.comments.replace_more(limit=0)
        top_comments = [comment.body for comment in post.comments.list()[:5]]

        posts_data.append({
            "subreddit": subreddit_name,
            "title": title,
            "content": content,
            "top_comments": top_comments,
            "universities": extracted_universities,
            "majors": extracted_majors
        })

    return pd.DataFrame(posts_data)

# 开始爬
reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                     client_secret=REDDIT_CLIENT_SECRET,
                     user_agent=REDDIT_USER_AGENT)

# 遍历目标，爬数据并合并
all_data = pd.concat([scrape_reddit_posts(sub) for sub in TARGET_SUBREDDITS], ignore_index=True)

# 导出结果
output_filename = "爬取的原始数据.csv"
all_data.to_csv(output_filename, index=False, encoding="utf-8")
print("爬取成功")