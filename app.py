import streamlit as st
import pandas as pd
import requests
import feedparser
from bs4 import BeautifulSoup

# NewsAPI.org 설정 (무료 플랜 사용)
NEWS_API_KEY = "d86964d07ac94c6d912dc929f6e2895e"  # NewsAPI.org에서 발급받은 API 키 입력
NEWS_API_URL = "https://newsapi.org/v2/everything"

# RSS 피드 URL 목록
RSS_FEEDS = [
    "https://www.yonhapnewstv.co.kr/category/headline/feed/",
    "https://www.khan.co.kr/rss/rssdata/total_news.xml",
    "https://www.donga.com/news/rss"
]

# 데이터프레임 초기화
news_data = pd.DataFrame(columns=["Date", "Source", "Title", "Link"])


# Streamlit 앱 시작
st.title("실시간 뉴스 타임라인")
st.write("계엄과 탄핵 관련 최신 뉴스를 한눈에 확인하세요.")

# --- 1. RSS 피드에서 데이터 수집 ---
def fetch_rss_data():
    global news_data
    for feed_url in RSS_FEEDS:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries:
            news_data = pd.concat([
                news_data,
                pd.DataFrame({
                    "Date": [entry.published[:10] if hasattr(entry, 'published') else "Unknown"],
                    "Source": [entry.link.split("/")[2]],
                    "Title": [entry.title],
                    "Link": [entry.link]
                })
            ], ignore_index=True)

# --- 2. NewsAPI.org에서 데이터 수집 ---
def fetch_newsapi_data():
    global news_data
    params = {
        "q": "계엄 OR 탄핵",  # 키워드 검색
        "apiKey": NEWS_API_KEY,
        "language": "ko",
        "sortBy": "publishedAt",
        "pageSize": 20
    }
    response = requests.get(NEWS_API_URL, params=params)
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        for article in articles:
            news_data = pd.concat([
                news_data,
                pd.DataFrame({
                    "Date": [article["publishedAt"][:10]],
                    "Source": [article["source"]["name"]],
                    "Title": [article["title"]],
                    "Link": [article["url"]]
                })
            ], ignore_index=True)

# --- 3. 데이터 수집 및 정리 ---
@st.cache_data
def get_news_data():
    fetch_rss_data()
    fetch_newsapi_data()
    news_data.sort_values("Date", ascending=False, inplace=True)  # 날짜별 정렬
    return news_data

news_data = get_news_data()

# --- 4. 사용자 필터링 기능 ---
st.sidebar.header("필터 옵션")
selected_date = st.sidebar.date_input("날짜 선택")
if selected_date:
    filtered_data = news_data[news_data["Date"] == str(selected_date)]
else:
    filtered_data = news_data

selected_keyword = st.sidebar.text_input("키워드 검색")
if selected_keyword:
    filtered_data = filtered_data[filtered_data["Title"].str.contains(selected_keyword, case=False)]

# --- 5. 데이터 출력 ---
st.subheader("뉴스 타임라인")
if not filtered_data.empty:
    for _, row in filtered_data.iterrows():
        st.markdown(f"- **{row['Title']}** ([출처: {row['Source']}]) [{row['Date']}]")
        st.write(f"[기사 보기]({row['Link']})")
else:
    st.write("선택한 조건에 해당하는 뉴스가 없습니다.")

# --- 6. 데이터 다운로드 기능 ---
st.sidebar.download_button(
    label="뉴스 데이터 다운로드 (CSV)",
    data=news_data.to_csv(index=False),
    file_name="news_data.csv",
    mime="text/csv"
)



# NewsAPI.org에서 데이터 수집
def fetch_newsapi_data():
    global news_data
    params = {
        "q": "계엄 OR 탄핵",  # 키워드 검색
        "apiKey": NEWS_API_KEY,
        "language": "ko",
        "sortBy": "publishedAt",
        "pageSize": 20
    }
    response = requests.get(NEWS_API_URL, params=params)
    if response.status_code == 200:
        articles = response.json().get("articles", [])
        for article in articles:
            news_data = pd.concat([
                news_data,
                pd.DataFrame({
                    "Date": [article.get("publishedAt", "Unknown")[:10]],
                    "Source": [article["source"]["name"]],
                    "Title": [article["title"]],
                    "Link": [article["url"]]
                })
            ], ignore_index=True)
    else:
        st.error(f"NewsAPI 오류: {response.status_code}")
        st.write(response.json())  # 디버깅용 출력

# RSS 피드에서 데이터 수집
def fetch_rss_data():
    global news_data
    for feed_url in RSS_FEEDS:
        feed = feedparser.parse(feed_url)
        for entry in feed.entries:
            news_data = pd.concat([
                news_data,
                pd.DataFrame({
                    "Date": [entry.get("published", "Unknown")[:10]],
                    "Source": [entry.link.split("/")[2]],
                    "Title": [entry.title],
                    "Link": [entry.link]
                })
            ], ignore_index=True)


# 데이터 수집 및 정리
def get_news_data():
    fetch_rss_data()
    fetch_newsapi_data()
    news_data.drop_duplicates(subset=["Title", "Link"], inplace=True)  # 중복 제거
    news_data.sort_values("Date", ascending=False, inplace=True)  # 날짜별 정렬
    return news_data

news_data = get_news_data()
st.write(news_data.head())  # 디버깅: 데이터 출력


