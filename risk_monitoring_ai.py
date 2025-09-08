"""
AI 기반 글로벌 리스크 모니터링 시스템 - Complete Enhanced Version
22개국 실시간 뉴스 모니터링 with AI 분석, 스케줄링, 캐싱
"""

import os
import json
import smtplib
import logging
import hashlib
import pickle
import schedule
import sys
import re
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dateutil import parser
from dateutil.relativedelta import relativedelta
from dataclasses import dataclass, asdict
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# .env 파일 로드
load_dotenv()

# 로그 설정
def setup_logging(log_level='INFO'):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    
    file_handler = logging.FileHandler(
        f'monitoring_{datetime.now().strftime("%Y%m%d")}.log', 
        encoding='utf-8'
    )
    file_handler.setFormatter(logging.Formatter(log_format))
    
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level))
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger

logger = setup_logging(os.getenv('LOG_LEVEL', 'INFO'))

@dataclass
class NewsItem:
    """News item data class"""
    title: str
    date: str
    source: str
    snippet: str
    link: str
    country: str
    country_code: str
    country_ko: str = ""
    thumbnail: str = ""
    search_type: str = "news"
    collected_at: str = ""
    news_hash: str = ""  # 뉴스 고유 해시
    
    # AI analysis result fields
    risk_score: float = 0.0
    risk_level: str = ""
    risk_category: str = ""
    ai_summary_ko: str = ""
    ai_full_translation_ko: str = ""
    is_duplicate: bool = False
    duplicate_of: str = ""
    ai_analysis_timestamp: str = ""
    
    def __post_init__(self):
        """뉴스 해시 생성"""
        if not self.news_hash:
            content = f"{self.title}{self.snippet}{self.source}"
            self.news_hash = hashlib.md5(content.encode()).hexdigest()

class NewsCache:
    """뉴스 캐시 관리 클래스"""
    
    def __init__(self, cache_dir: str = "news_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.company_cache_file = self.cache_dir / "company_news_cache.pkl"
        self.daily_cache_file = self.cache_dir / "daily_news_cache.pkl"
        
    def load_company_cache(self) -> Set[str]:
        """회사 관련 뉴스 캐시 로드"""
        if self.company_cache_file.exists():
            try:
                with open(self.company_cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return set()
        return set()
    
    def save_company_cache(self, news_hashes: Set[str]):
        """회사 관련 뉴스 캐시 저장"""
        with open(self.company_cache_file, 'wb') as f:
            pickle.dump(news_hashes, f)
    
    def clear_daily_cache(self):
        """일일 캐시 초기화"""
        if self.daily_cache_file.exists():
            self.daily_cache_file.unlink()

class GeminiAnalyzer:
    """Gemini AI 분석 클래스"""
    
    def __init__(self, api_key: str):
        """Gemini API 초기화"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        logger.info("✅ Gemini 2.0 Flash 초기화 완료")
        
        # 리스크 점수 기준
        self.risk_thresholds = {
            'HIGH': 70,     # 70점 이상
            'MEDIUM': 40,   # 40-69점
            'LOW': 20       # 20-39점
        }
    
    def remove_duplicates(self, news_list: List[NewsItem]) -> List[NewsItem]:
        """AI 기반 중복 뉴스 제거"""
        logger.info("🔍 AI 기반 중복 뉴스 제거 시작...")
        
        if not news_list:
            return []
        
        unique_news = []
        
        # 국가별로 그룹화
        country_news = {}
        for news in news_list:
            country = news.country
            if country not in country_news:
                country_news[country] = []
            country_news[country].append(news)
        
        # 각 국가별로 AI 중복 체크
        for country, items in country_news.items():
            if not items:
                continue
            
            # 날짜순 정렬 (최신 우선)
            items.sort(key=lambda x: x.date, reverse=True)
            
            # 첫 번째 뉴스는 무조건 포함
            unique_news.append(items[0])
            
            # 나머지 뉴스들에 대해 중복 체크
            for i in range(1, len(items)):
                candidate = items[i]
                
                # 이미 추가된 뉴스들과 비교 (같은 국가 내에서만)
                is_duplicate = False
                duplicate_of = None
                
                # 배치로 비교 (최대 5개씩)
                country_unique = [n for n in unique_news if n.country == country]
                
                if country_unique:
                    # AI로 중복 판단
                    is_duplicate, duplicate_of = self._check_duplicate_with_ai(
                        candidate, 
                        country_unique[-min(5, len(country_unique)):]  # 최근 5개와만 비교
                    )
                
                if not is_duplicate:
                    unique_news.append(candidate)
                else:
                    candidate.is_duplicate = True
                    candidate.duplicate_of = duplicate_of or ""
            
            # 진행 상황 로그
            logger.info(f"  - {country}: {len(items)}건 → {len([n for n in unique_news if n.country == country])}건")
        
        logger.info(f"✅ AI 중복 제거 완료: {len(news_list)} → {len(unique_news)}건")
        return unique_news
    
    def _check_duplicate_with_ai(self, candidate: NewsItem, existing_news: List[NewsItem]) -> Tuple[bool, Optional[str]]:
        """AI를 사용한 중복 체크"""
        try:
            prompt = f"""Determine if the candidate news article covers the same event as any of the existing news articles.

[CANDIDATE NEWS]
Title: {candidate.title}
Content: {candidate.snippet}
Date: {candidate.date}
Source: {candidate.source}

[EXISTING NEWS LIST]
"""
            for idx, news in enumerate(existing_news, 1):
                prompt += f"""
{idx}.
Title: {news.title}
Content: {news.snippet}
Date: {news.date}
Source: {news.source}
"""
            
            prompt += """

CRITERIA FOR JUDGMENT:
1. Do they cover the same incident/event?
2. Do the key details (location, time, casualty numbers, etc.) match?
3. Is this actually the same news, not just a similar topic?

RESPONSE FORMAT:
IsDuplicate: (Yes/No)
DuplicateNumber: (1-5, or 0 if not duplicate)

Please respond concisely."""
            
            response = self.model.generate_content(prompt)
            result = response.text.strip()
            
            # Parse response
            is_duplicate = False
            duplicate_idx = 0
            
            lines = result.split('\n')
            for line in lines:
                line_lower = line.lower()
                if 'isduplicate:' in line_lower and 'yes' in line_lower:
                    is_duplicate = True
                elif 'duplicatenumber:' in line_lower:
                    match = re.search(r'\d+', line)
                    if match:
                        duplicate_idx = int(match.group())
            
            if is_duplicate and 1 <= duplicate_idx <= len(existing_news):
                return True, existing_news[duplicate_idx - 1].link
            
            return False, None
            
        except Exception as e:
            logger.error(f"AI duplicate check error: {e}")
            return False, None
    
    def analyze_risk_batch(self, news_list: List[NewsItem], batch_size: int = 5) -> List[NewsItem]:
        """배치 단위로 리스크 분석"""
        logger.info(f"🤖 AI 리스크 분석 시작 ({len(news_list)}건)...")
        
        analyzed_news = []
        
        # 배치 처리
        for i in range(0, len(news_list), batch_size):
            batch = news_list[i:i+batch_size]
            
            # 프롬프트 생성
            prompt = self._create_risk_analysis_prompt(batch)
            
            try:
                # Gemini API 호출
                response = self.model.generate_content(prompt)
                
                # 응답 파싱
                results = self._parse_risk_response(response.text, batch)
                analyzed_news.extend(results)
                
                # API 호출 간격
                time.sleep(1)
                
                # 진행상황 로그
                logger.info(f"  - 분석 진행: {min(i+batch_size, len(news_list))}/{len(news_list)}")
                
            except Exception as e:
                logger.error(f"❌ AI 분석 오류: {e}")
                # 오류 시 기본값 설정
                for news in batch:
                    news.risk_score = 0
                    news.risk_level = ""
                analyzed_news.extend(batch)
        
        # 리스크 점수 기준으로 필터링
        filtered_news = [n for n in analyzed_news if n.risk_score >= self.risk_thresholds['LOW']]
        
        logger.info(f"✅ AI 분석 완료: {len(filtered_news)}건이 리스크 기준 충족")
        return filtered_news
    
    def _create_risk_analysis_prompt(self, news_batch: List[NewsItem]) -> str:
        """리스크 분석 프롬프트 생성"""
        prompt = """You are a risk analysis expert for a global construction company.
Please analyze the following news articles and evaluate the risk score for each.

EVALUATION CRITERIA:
1. Business Impact (0-40 points)
   - Project disruption/delay possibility
   - Financial loss magnitude
   - Legal/regulatory risks
   
2. Reputation Impact (0-30 points)
   - Negative media coverage potential
   - Brand image damage level
   - Stakeholder trust impact
   
3. Employee Safety/Harm (0-30 points)
   - Employee life/safety threats
   - Work environment deterioration
   - Evacuation/withdrawal necessity

SPECIAL WEIGHTS:
- Direct mention of Samsung C&T or Samsung Construction: +20 points
- Accidents with 10+ fatalities: +30 points
- National-scale disasters/calamities: +25 points
- Large-scale protests/political instability: +20 points

If news source is Samsung or official Samsung channels: -50 points (minimum 0 points)

DATE RELEVANCE PENALTY:
- News older than 7 days: -30 points
- News older than 30 days: -50 points
- News older than 1 year: -80 points

For each news item, respond in the following format:
[News Number]
RiskScore: (0-100)
RiskCategory: (Natural Disaster/Political Unrest/Accident/Health Crisis/Economic Crisis/Other)
KeyRisk: (One sentence summary)

NEWS LIST:
"""
        
        for idx, news in enumerate(news_batch):
            prompt += f"\n[{idx+1}]\n"
            prompt += f"Title: {news.title}\n"
            prompt += f"Country: {news.country}\n"
            prompt += f"Source: {news.source}\n"
            prompt += f"Content: {news.snippet}\n"
            prompt += f"Date: {news.date}\n"
        
        return prompt
    
    def _parse_risk_response(self, response_text: str, news_batch: List[NewsItem]) -> List[NewsItem]:
        """AI 응답 파싱"""
        results = []
        
        # 응답을 뉴스별로 분리
        sections = response_text.split('[')
        
        for section in sections[1:]:  # 첫 번째는 빈 문자열
            try:
                lines = section.strip().split('\n')
                
                # 뉴스 번호 추출
                news_idx = int(lines[0].split(']')[0]) - 1
                if news_idx >= len(news_batch):
                    continue
                
                news = news_batch[news_idx]
                
                # 리스크 정보 파싱
                for line in lines[1:]:
                    line_lower = line.lower()
                    if 'riskscore:' in line_lower:
                        score_match = re.findall(r'\d+', line)
                        if score_match:
                            score = float(score_match[0])
                            news.risk_score = score
                    elif 'riskcategory:' in line_lower:
                        parts = line.split(':', 1)
                        if len(parts) > 1:
                            category = parts[1].strip()
                            news.risk_category = category
                
                # 리스크 레벨 결정
                if news.risk_score >= self.risk_thresholds['HIGH']:
                    news.risk_level = 'HIGH'
                elif news.risk_score >= self.risk_thresholds['MEDIUM']:
                    news.risk_level = 'MEDIUM'
                elif news.risk_score >= self.risk_thresholds['LOW']:
                    news.risk_level = 'LOW'
                else:
                    news.risk_level = ''
                
                news.ai_analysis_timestamp = datetime.now().isoformat()
                results.append(news)
                
            except Exception as e:
                logger.error(f"파싱 오류: {e}")
                continue
        
        return results
    
    def summarize_and_translate(self, news_list: List[NewsItem]) -> List[NewsItem]:
        """뉴스 요약 및 한국어 번역"""
        logger.info("📝 뉴스 요약 및 번역 시작...")
        
        total_items = len([n for n in news_list if n.risk_level])
        processed = 0
        
        for news in news_list:
            if not news.risk_level:
                continue
            
            try:
                if news.risk_level == 'HIGH':
                    # HIGH: 전체 번역 + 요약
                    prompt = f"""Please translate and summarize the following news into Korean.

Title: {news.title}
Content: {news.snippet}
Date: {news.date}
Country: {news.country}

Please respond in the following format:
[Summary]
(3-4 sentences summarizing key points in Korean)

[Full Translation]
(Complete translation of the content in natural Korean)"""
                    
                    response = self.model.generate_content(prompt)
                    result = response.text
                    
                    # Separate summary and translation
                    if '[Summary]' in result and '[Full Translation]' in result:
                        summary = result.split('[Summary]')[1].split('[Full Translation]')[0].strip()
                        translation = result.split('[Full Translation]')[1].strip()
                        news.ai_summary_ko = summary
                        news.ai_full_translation_ko = translation
                    
                else:
                    # MEDIUM/LOW: Summary only
                    prompt = f"""Please summarize the following news in 3-4 sentences in Korean.

Title: {news.title}
Content: {news.snippet}
Date: {news.date}
Country: {news.country}"""
                    
                    response = self.model.generate_content(prompt)
                    news.ai_summary_ko = response.text.strip()
                
                time.sleep(0.5)  # API 호출 간격
                
                processed += 1
                if processed % 10 == 0:
                    logger.info(f"  - 번역 진행: {processed}/{total_items}")
                
            except Exception as e:
                logger.error(f"번역/요약 오류: {e}")
                news.ai_summary_ko = "Translation failed"
        
        logger.info("✅ 요약 및 번역 완료")
        return news_list

class AIRiskMonitoringSystem:
    """AI 기반 리스크 모니터링 시스템 (기본)"""
    
    def __init__(self, config_path='monitoring_config.json'):
        """시스템 초기화"""
        logger.info("="*70)
        logger.info("🤖 AI 기반 글로벌 리스크 모니터링 시스템 초기화")
        logger.info("="*70)
        
        # API 키 확인
        self.serpapi_key = os.getenv('SERPAPI_KEY')
        self.gemini_key = os.getenv('GEMINI_API_KEY')
        
        if not self.serpapi_key:
            logger.error("❌ SERPAPI_KEY가 설정되지 않았습니다.")
            sys.exit(1)
        
        if not self.gemini_key:
            logger.error("❌ GEMINI_API_KEY가 설정되지 않았습니다.")
            sys.exit(1)
        
        # SerpAPI 초기화
        try:
            from serpapi import GoogleSearch
            self.GoogleSearch = GoogleSearch
            logger.info("✅ SerpAPI 패키지 로드 완료")
        except ImportError:
            logger.error("❌ serpapi 패키지가 설치되지 않았습니다.")
            sys.exit(1)
        
        # Gemini 분석기 초기화
        self.analyzer = GeminiAnalyzer(self.gemini_key)
        
        # 설정 파일 로드
        self.load_config(config_path)
        
        # 이메일 설정
        self.setup_email_config()
        
        # Initialize statistics
        self.stats = {
            'api_calls': 0,
            'news_collected': 0,
            'news_after_dedup': 0,
            'news_analyzed': 0,
            'high_risk': 0,
            'medium_risk': 0,
            'low_risk': 0,
            'errors': 0,
            'start_time': datetime.now()
        }
    
    def load_config(self, config_path):
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            self.countries = {
                k: v for k, v in self.config['countries'].items() 
                if v.get('active', True)
            }
            
            self.risk_keywords = self.config['search_keywords']['risk_keywords']
            self.combined_query = self.config['search_keywords']['combined_query']
            self.company_keywords = self.config['company_keywords']
            self.korean_media = self.config.get('korean_media', {})
            
            logger.info(f"✅ 설정 파일 로드 완료: {config_path}")
            
        except Exception as e:
            logger.error(f"❌ 설정 파일 로드 실패: {e}")
            sys.exit(1)
    
    def setup_email_config(self):
        """이메일 설정"""
        self.email_config = {
            'smtp_server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': int(os.getenv('SMTP_PORT', 587)),
            'sender_email': os.getenv('SENDER_EMAIL', ''),
            'sender_password': os.getenv('SENDER_PASSWORD', ''),
            'recipients': []
        }
        
        env_recipients = os.getenv('RECIPIENT_EMAILS', '')
        if env_recipients:
            self.email_config['recipients'] = [
                email.strip() for email in env_recipients.split(',')
            ]

    def parse_news_date(self, date_str: str) -> Optional[datetime]:
        """Parse various date formats from news"""
        if not date_str:
            return None
            
        try:
            # Remove timezone info and clean up
            date_str = date_str.strip()
            
            # Try dateutil parser first (handles many formats)
            parsed_date = parser.parse(date_str, fuzzy=True)
            return parsed_date
            
        except Exception as e:
            logger.debug(f"Date parsing failed for: {date_str} - {e}")
            
            # Try specific formats as fallback
            formats = [
                '%m/%d/%Y',
                '%Y-%m-%d',
                '%d/%m/%Y',
                '%B %d, %Y',
                '%b %d, %Y',
                '%Y/%m/%d',
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(date_str.split(',')[0], fmt)
                except:
                    continue
            
            # If all parsing fails, return None
            return None
        
    def filter_recent_news(self, news_list: List[NewsItem], days: int = 7) -> List[NewsItem]:
        """Filter news to keep only recent items within specified days"""
        logger.info(f"📅 Starting date filtering (keeping last {days} days)...")
        
        # timezone-aware datetime 생성
        from datetime import timezone
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        filtered_news = []
        old_news_count = 0
        unparseable_dates = 0
        
        for news in news_list:
            # Parse the news date
            news_date = self.parse_news_date(news.date)
            
            if news_date is None:
                # 날짜를 파싱할 수 없으면 포함하되 경고 로그
                unparseable_dates += 1
                logger.warning(f"Cannot parse date for: {news.title[:50]}... Date: {news.date}")
                filtered_news.append(news)
                continue
            
            # timezone이 없는 날짜에 UTC timezone 추가
            if news_date.tzinfo is None:
                news_date = news_date.replace(tzinfo=timezone.utc)
            
            # 뉴스가 최근 것인지 확인
            if news_date >= cutoff_date:
                filtered_news.append(news)
            else:
                old_news_count += 1
                # UTC 기준으로 날짜 차이 계산
                current_time = datetime.now(timezone.utc)
                days_old = (current_time - news_date).days
                logger.debug(f"Filtered old news ({days_old} days old): {news.title[:50]}...")
        
        logger.info(f"✅ Date filtering complete: {len(news_list)} → {len(filtered_news)} items")
        logger.info(f"   - Removed {old_news_count} old news items")
        if unparseable_dates > 0:
            logger.warning(f"   - {unparseable_dates} items with unparseable dates (kept)")
        
        return filtered_news

    def search_news(self, query: str, country_code: str = None, 
                   country_name: str = None, search_type: str = 'news') -> List[NewsItem]:
        """뉴스 검색"""
        results = []
        
        try:
            if search_type == 'news':
                params = {
                    "api_key": self.serpapi_key,
                    "engine": "google_news",
                    "q": query,
                    "when": "7d"
                }
                
                if country_code:
                    params["gl"] = country_code
                    params["hl"] = "en"
            else:
                params = {
                    "api_key": self.serpapi_key,
                    "engine": "google",
                    "q": query,
                    "num": "10",
                    "tbs": "qdr:w"
                }
                
                if country_code:
                    params["gl"] = country_code
                    params["hl"] = "ko" if country_code == "kr" else "en"
            
            search = self.GoogleSearch(params)
            response = search.get_dict()
            
            self.stats['api_calls'] += 1
            
            # 결과 파싱
            if search_type == 'news' and "news_results" in response:
                for item in response["news_results"][:10]:
                    news_item = NewsItem(
                        title=item.get('title', ''),
                        date=item.get('date', ''),
                        source=item.get('source', {}).get('name', 'Unknown'),
                        snippet=item.get('snippet', ''),
                        link=item.get('link', ''),
                        country=country_name or 'Global',
                        country_code=country_code or 'global',
                        thumbnail=item.get('thumbnail', ''),
                        search_type=search_type,
                        collected_at=datetime.now().isoformat()
                    )
                    results.append(news_item)
                    
            elif search_type != 'news' and "organic_results" in response:
                for item in response["organic_results"][:5]:
                    news_item = NewsItem(
                        title=item.get('title', ''),
                        date=datetime.now().strftime('%Y-%m-%d'),
                        source=item.get('displayed_link', 'Unknown'),
                        snippet=item.get('snippet', ''),
                        link=item.get('link', ''),
                        country=country_name or 'Korea',
                        country_code=country_code or 'kr',
                        search_type=search_type,
                        collected_at=datetime.now().isoformat()
                    )
                    results.append(news_item)
            
            self.stats['news_collected'] += len(results)
            
        except Exception as e:
            logger.error(f"API 오류: {e}")
            self.stats['errors'] += 1
        
        return results
    
    def collect_all_news(self) -> List[NewsItem]:
        """모든 뉴스 수집"""
        all_news = []
        
        # 1. 국가별 뉴스
        logger.info("\n🌍 국가별 리스크 뉴스 수집 시작")
        for idx, (country_code, country_info) in enumerate(self.countries.items(), 1):
            logger.info(f"[{idx}/{len(self.countries)}] {country_info['name_ko']} ({country_info['name']})")
            
            query = f"{country_info['name']} {self.combined_query}"
            news = self.search_news(
                query=query,
                country_code=country_info['gl'],
                country_name=country_info['name'],
                search_type='news'
            )
            
            for item in news:
                item.country_ko = country_info['name_ko']
            
            all_news.extend(news)
            time.sleep(1)
        
        # 2. 회사 관련 뉴스
        logger.info("\n🏢 회사 키워드 뉴스 수집 시작")
        for idx, keyword in enumerate(self.company_keywords, 1):
            logger.info(f"[{idx}/{len(self.company_keywords)}] {keyword}")
            
            query = f'"{keyword}" construction project accident'
            news = self.search_news(query=query, search_type='news')
            all_news.extend(news)
            time.sleep(1)
        
        # 3. 한국 미디어
        logger.info("\n🇰🇷 한국 언론 모니터링")
        for site in self.korean_media.get('sites', []):
            if not site.get('active', False):
                continue
            
            for term in self.korean_media.get('search_terms', []):
                query = f'{site["selector"]} "{term}"'
                news = self.search_news(
                    query=query,
                    country_code='kr',
                    country_name='Korea',
                    search_type='web'
                )
                all_news.extend(news)
                time.sleep(1)
        
        # 4. 오래된 뉴스 삭제를 위한 날짜 필터링 적용
        logger.info(f"\n📅 날짜 필터링 적용 중...")
        logger.info(f"필터링 이전 전체 뉴스 갯수 : {len(all_news)}")
        
        # Filter to keep only recent news (default: last 7 days)
        days_to_keep = self.config.get('search_settings', {}).get('days_to_keep', 7)
        all_news = self.filter_recent_news(all_news, days=days_to_keep)
        
        logger.info(f"필터링 이후 전체 뉴스 갯수 : {len(all_news)}")
        
        return all_news
    
    def create_ai_html_report(self, analyzed_news: List[NewsItem]) -> str:
        """AI 분석 결과를 포함한 HTML 리포트 생성"""
        # 리스크 레벨별 분류
        high_risk = [n for n in analyzed_news if n.risk_level == 'HIGH']
        medium_risk = [n for n in analyzed_news if n.risk_level == 'MEDIUM']
        low_risk = [n for n in analyzed_news if n.risk_level == 'LOW']
        
        html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>글로벌 리스크 모니터링 리포트 - {datetime.now().strftime('%Y-%m-%d')}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans KR', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 36px;
            margin-bottom: 10px;
            font-weight: 700;
        }}
        .ai-badge {{
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 5px 15px;
            border-radius: 20px;
            margin-top: 10px;
        }}
        .stats-container {{
            background: #f8f9fa;
            padding: 30px;
            border-bottom: 2px solid #e9ecef;
        }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
        }}
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 12px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }}
        .content {{
            padding: 40px;
        }}
        .news-card {{
            background: white;
            border: 1px solid #e9ecef;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 20px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        .news-card.high-risk {{
            border-left: 5px solid #dc3545;
        }}
        .news-card.medium-risk {{
            border-left: 5px solid #ffc107;
        }}
        .news-card.low-risk {{
            border-left: 5px solid #28a745;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌍 글로벌 리스크 모니터링 리포트</h1>
            <div class="ai-badge">Powered by Gemini 2.0 Flash</div>
        </div>
        
        <div class="stats-container">
            <div class="stats-grid">
                <div class="stat-card">
                    <div style="font-size: 42px; font-weight: bold;">{len(analyzed_news)}</div>
                    <div>Total News</div>
                </div>
                <div class="stat-card">
                    <div style="font-size: 42px; font-weight: bold; color: #dc3545;">{len(high_risk)}</div>
                    <div>High Risk</div>
                </div>
                <div class="stat-card">
                    <div style="font-size: 42px; font-weight: bold; color: #ffc107;">{len(medium_risk)}</div>
                    <div>Medium Risk</div>
                </div>
                <div class="stat-card">
                    <div style="font-size: 42px; font-weight: bold; color: #28a745;">{len(low_risk)}</div>
                    <div>Low Risk</div>
                </div>
            </div>
        </div>
        
        <div class="content">
"""
        
        # HIGH RISK 섹션
        if high_risk:
            html += "<h2>⚠️ HIGH RISK - 즉시 확인 필요</h2>"
            for news in high_risk[:20]:
                html += self._create_ai_news_card(news, 'high')
        
        # MEDIUM RISK 섹션
        if medium_risk:
            html += "<h2>📢 MEDIUM RISK - 주의 필요</h2>"
            for news in medium_risk[:15]:
                html += self._create_ai_news_card(news, 'medium')
        
        # LOW RISK 섹션
        if low_risk:
            html += "<h2>ℹ️ LOW RISK - 모니터링</h2>"
            for news in low_risk[:10]:
                html += self._create_ai_news_card(news, 'low')
        
        html += """
        </div>
    </div>
</body>
</html>"""
        
        return html
    
    def _create_ai_news_card(self, news: NewsItem, risk_class: str) -> str:
        """뉴스 카드 HTML 생성"""
        import html
        
        return f"""
        <div class="news-card {risk_class}-risk">
            <h3>{html.escape(news.title)}</h3>
            <p>📍 {news.country_ko or news.country} | 📰 {html.escape(news.source)} | 📅 {news.date}</p>
            <p><strong>리스크 점수:</strong> {news.risk_score:.0f} | <strong>카테고리:</strong> {news.risk_category or 'Other'}</p>
            <p><strong>AI 요약:</strong> {html.escape(news.ai_summary_ko or 'No summary')}</p>
            <a href="{news.link}" target="_blank">원문 보기 →</a>
        </div>
"""

    def send_email_report(self, html_content: str, news_list: List[NewsItem], use_email_version: bool = False) -> bool:
        """이메일로 리포트 전송"""
        try:
            # 고위험 뉴스 개수
            high_risk_count = len([n for n in news_list if n.risk_level == 'HIGH'])
            
            # 이메일 제목
            subject = f"[리스크 모니터링] {datetime.now().strftime('%Y-%m-%d')} - "
            if high_risk_count > 0:
                subject += f"⚠️ HIGH RISK {high_risk_count}건 발생"
            else:
                subject += "정상 모니터링 완료"
            
            # 이메일 생성
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_config['sender_email']
            msg['To'] = ', '.join(self.email_config['recipients'])
            
            # HTML 본문 첨부
            html_part = MIMEText(html_content, 'html', 'utf-8')
            msg.attach(html_part)
            
            # SMTP 서버 연결 및 전송
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['sender_email'], self.email_config['sender_password'])
                server.send_message(msg)
                
            logger.info(f"📧 이메일 전송 성공: {', '.join(self.email_config['recipients'])}")
            return True
            
        except Exception as e:
            logger.error(f"이메일 전송 실패: {e}")
            return False

    def run(self):
        """메인 실행 함수"""
        logger.info("\n" + "="*70)
        logger.info("🚀 AI 기반 글로벌 리스크 모니터링 시작")
        logger.info("="*70)
        
        try:
            # 1. 뉴스 수집
            logger.info("\n📡 뉴스 수집 단계")
            all_news = self.collect_all_news()
            logger.info(f"✅ 총 {len(all_news)}건 수집 완료")
            
            # 2. AI 기반 중복 제거
            logger.info("\n🔍 AI 중복 제거 단계")
            unique_news = self.analyzer.remove_duplicates(all_news)
            self.stats['news_after_dedup'] = len(unique_news)
            logger.info(f"✅ 중복 제거 후 {len(unique_news)}건")
            
            # 3. AI 리스크 분석
            logger.info("\n🤖 AI 리스크 분석 단계")
            analyzed_news = self.analyzer.analyze_risk_batch(unique_news)
            self.stats['news_analyzed'] = len(analyzed_news)
            
            # 4. 요약 및 번역
            logger.info("\n📝 요약 및 번역 단계")
            final_news = self.analyzer.summarize_and_translate(analyzed_news)
            
            # 5. 통계 업데이트
            self.stats['high_risk'] = len([n for n in final_news if n.risk_level == 'HIGH'])
            self.stats['medium_risk'] = len([n for n in final_news if n.risk_level == 'MEDIUM'])
            self.stats['low_risk'] = len([n for n in final_news if n.risk_level == 'LOW'])
            
            # 6. 리포트 생성
            logger.info("\n📊 리포트 생성 단계")
            html_content = self.create_ai_html_report(final_news)
            
            # 7. 파일 저장
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            html_file = f'ai_risk_report_{timestamp}.html'
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # 8. 이메일 전송
            if self.email_config['sender_email'] and self.email_config['recipients']:
                logger.info("\n📧 이메일 전송 시작...")
                email_sent = self.send_email_report(html_content, final_news)
                if email_sent:
                    logger.info("✅ 이메일 전송 완료")
                else:
                    logger.error("❌ 이메일 전송 실패")
            
            # 9. 결과 출력
            duration = datetime.now() - self.stats['start_time']
            logger.info("\n" + "="*70)
            logger.info("✅ AI 리스크 모니터링 완료!")
            logger.info(f"소요 시간: {str(duration).split('.')[0]}")
            logger.info(f"수집: {self.stats['news_collected']}건")
            logger.info(f"AI 분석: {self.stats['news_analyzed']}건")
            logger.info(f"HIGH: {self.stats['high_risk']}건")
            logger.info(f"MEDIUM: {self.stats['medium_risk']}건")
            logger.info(f"LOW: {self.stats['low_risk']}건")
            logger.info(f"생성 파일: {html_file}")
            logger.info("="*70)
            
            return {
                'success': True,
                'stats': self.stats,
                'files': {'html': html_file}
            }
            
        except Exception as e:
            logger.error(f"❌ 실행 중 오류: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {'success': False, 'error': str(e)}

class EnhancedAIRiskMonitoringSystem(AIRiskMonitoringSystem):
    """개선된 AI 리스크 모니터링 시스템"""
    
    def __init__(self, config_path='monitoring_config.json', mode='normal'):
        """
        시스템 초기화
        mode: 'normal' (정상 실행), 'test' (테스트 - 1회 실행), 'schedule' (스케줄링)
        """
        super().__init__(config_path)
        self.mode = mode
        self.news_cache = NewsCache()
        self.company_news_hashes = self.news_cache.load_company_cache()
        
        logger.info(f"🚀 시스템 모드: {mode}")
        
    def collect_company_news(self) -> List[NewsItem]:
        """회사 관련 뉴스만 수집"""
        all_news = []
        
        # 1. 회사 키워드 뉴스
        logger.info("\n🏢 회사 키워드 뉴스 수집 시작")
        for idx, keyword in enumerate(self.company_keywords, 1):
            logger.info(f"[{idx}/{len(self.company_keywords)}] {keyword}")
            
            query = f'"{keyword}" construction project accident'
            news = self.search_news(query=query, search_type='news')
            
            # 국가를 "삼성물산"으로 설정
            for item in news:
                item.country = "삼성물산"
                item.country_ko = "삼성물산"
                item.country_code = "samsung"
            
            all_news.extend(news)
            time.sleep(1)
        
        # 2. 한국 미디어에서 삼성물산 검색
        logger.info("\n🇰🇷 한국 언론 모니터링")
        for site in self.korean_media.get('sites', []):
            if not site.get('active', False):
                continue
            
            for term in self.korean_media.get('search_terms', []):
                query = f'{site["selector"]} "{term}"'
                news = self.search_news(
                    query=query,
                    country_code='kr',
                    country_name='삼성물산',  # Korea 대신 삼성물산
                    search_type='web'
                )
                
                # 국가를 "삼성물산"으로 설정
                for item in news:
                    item.country = "삼성물산"
                    item.country_ko = "삼성물산"
                    item.country_code = "samsung"
                
                all_news.extend(news)
                time.sleep(1)
        
        return all_news
    
    def collect_country_news(self) -> List[NewsItem]:
        """국가별 리스크 뉴스만 수집"""
        all_news = []
        
        logger.info("\n🌍 국가별 리스크 뉴스 수집 시작")
        for idx, (country_code, country_info) in enumerate(self.countries.items(), 1):
            logger.info(f"[{idx}/{len(self.countries)}] {country_info['name_ko']} ({country_info['name']})")
            
            query = f"{country_info['name']} {self.combined_query}"
            news = self.search_news(
                query=query,
                country_code=country_info['gl'],
                country_name=country_info['name'],
                search_type='news'
            )
            
            for item in news:
                item.country_ko = country_info['name_ko']
            
            all_news.extend(news)
            time.sleep(1)
        
        return all_news
    
    def filter_new_company_news(self, news_list: List[NewsItem]) -> List[NewsItem]:
        """새로운 회사 뉴스만 필터링"""
        new_news = []
        new_hashes = set()
        
        for news in news_list:
            if news.news_hash not in self.company_news_hashes:
                new_news.append(news)
                new_hashes.add(news.news_hash)
                logger.info(f"🆕 새로운 뉴스 발견: {news.title[:50]}...")
        
        # 캐시 업데이트
        if new_hashes:
            self.company_news_hashes.update(new_hashes)
            self.news_cache.save_company_cache(self.company_news_hashes)
            logger.info(f"✅ {len(new_news)}건의 새로운 회사 뉴스 발견")
        else:
            logger.info("ℹ️ 새로운 회사 뉴스 없음")
        
        return new_news
    
    def create_email_compatible_html_report(self, analyzed_news: List[NewsItem]) -> str:
        """이메일 클라이언트 호환 HTML 리포트 생성"""
        
        # 리스크 레벨별 분류
        high_risk = [n for n in analyzed_news if n.risk_level == 'HIGH']
        medium_risk = [n for n in analyzed_news if n.risk_level == 'MEDIUM']
        low_risk = [n for n in analyzed_news if n.risk_level == 'LOW']
        
        # 국가별 리스크 집계
        country_risks = {}
        for news in analyzed_news:
            country = news.country_ko or news.country
            if country not in country_risks:
                country_risks[country] = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0}
            country_risks[country][news.risk_level] += 1
        
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>리스크 모니터링 리포트 - {datetime.now().strftime('%Y-%m-%d')}</title>
</head>
<body style="margin: 0; padding: 0; font-family: Arial, sans-serif; background-color: #f4f4f4;">
    
    <!-- 전체 컨테이너 -->
    <table width="100%" cellpadding="0" cellspacing="0" border="0" style="background-color: #f4f4f4;">
        <tr>
            <td align="center" style="padding: 20px;">
                
                <!-- 메인 컨테이너 -->
                <table width="600" cellpadding="0" cellspacing="0" border="0" style="background-color: #ffffff; border-radius: 8px;">
                    
                    <!-- 헤더 -->
                    <tr>
                        <td style="background-color: #6b46c1; padding: 30px; text-align: center; border-radius: 8px 8px 0 0;">
                            <h1 style="color: #ffffff; margin: 0; font-size: 28px;">🌍 G/O실 글로벌 리스크 모니터링</h1>
                            <p style="color: #ffffff; margin: 10px 0 0 0; font-size: 14px;">
                                {datetime.now().strftime('%Y년 %m월 %d일')} | Samsung C&T
                            </p>
                        </td>
                    </tr>
                    
                    <!-- 통계 요약 -->
                    <tr>
                        <td style="padding: 30px;">
                            <table width="100%" cellpadding="0" cellspacing="0" border="0">
                                <tr>
                                    <td style="text-align: center; padding: 10px;">
                                        <div style="font-size: 36px; font-weight: bold; color: #6b46c1;">{len(analyzed_news)}</div>
                                        <div style="color: #666; font-size: 12px; margin-top: 5px;">전체 뉴스</div>
                                    </td>
                                    <td style="text-align: center; padding: 10px;">
                                        <div style="font-size: 36px; font-weight: bold; color: #dc3545;">{len(high_risk)}</div>
                                        <div style="color: #666; font-size: 12px; margin-top: 5px;">HIGH RISK</div>
                                    </td>
                                    <td style="text-align: center; padding: 10px;">
                                        <div style="font-size: 36px; font-weight: bold; color: #ffc107;">{len(medium_risk)}</div>
                                        <div style="color: #666; font-size: 12px; margin-top: 5px;">MEDIUM RISK</div>
                                    </td>
                                    <td style="text-align: center; padding: 10px;">
                                        <div style="font-size: 36px; font-weight: bold; color: #28a745;">{len(low_risk)}</div>
                                        <div style="color: #666; font-size: 12px; margin-top: 5px;">LOW RISK</div>
                                    </td>
                                </tr>
                            </table>
                        </td>
                    </tr>
                    
                    <!-- 국가별 현황 -->
                    <tr>
                        <td style="padding: 0 30px 30px;">
                            <h2 style="color: #333; font-size: 20px; margin-bottom: 15px;">📊 국가별 리스크 현황</h2>
                            <table width="100%" cellpadding="8" cellspacing="0" border="1" style="border-collapse: collapse; border-color: #ddd;">
                                <tr style="background-color: #f8f9fa;">
                                    <th style="text-align: left; color: #333;">국가</th>
                                    <th style="text-align: center; color: #333;">HIGH</th>
                                    <th style="text-align: center; color: #333;">MEDIUM</th>
                                    <th style="text-align: center; color: #333;">LOW</th>
                                    <th style="text-align: center; color: #333;">총계</th>
                                </tr>"""
    
        # 국가별 데이터 추가
        for country, risks in sorted(country_risks.items(), 
                                    key=lambda x: (x[1]['HIGH'], x[1]['MEDIUM']), 
                                    reverse=True):
            total = risks['HIGH'] + risks['MEDIUM'] + risks['LOW']
            html += f"""
                                <tr>
                                    <td style="padding: 8px;"><strong>{country}</strong></td>
                                    <td style="text-align: center; padding: 8px;">
                                        {f'<span style="background-color: #dc3545; color: white; padding: 2px 8px; border-radius: 3px;">{risks["HIGH"]}</span>' if risks['HIGH'] > 0 else '-'}
                                    </td>
                                    <td style="text-align: center; padding: 8px;">
                                        {f'<span style="background-color: #ffc107; color: #333; padding: 2px 8px; border-radius: 3px;">{risks["MEDIUM"]}</span>' if risks['MEDIUM'] > 0 else '-'}
                                    </td>
                                    <td style="text-align: center; padding: 8px;">
                                        {f'<span style="background-color: #28a745; color: white; padding: 2px 8px; border-radius: 3px;">{risks["LOW"]}</span>' if risks['LOW'] > 0 else '-'}
                                    </td>
                                    <td style="text-align: center; padding: 8px;"><strong>{total}</strong></td>
                                </tr>"""
        
        html += """
                            </table>
                        </td>
                    </tr>"""
        
        # HIGH RISK 뉴스
        if high_risk:
            html += """
                    <tr>
                        <td style="padding: 0 30px 30px;">
                            <h2 style="color: #dc3545; font-size: 20px; margin-bottom: 15px;">⚠️ HIGH RISK - 즉시 확인 필요</h2>"""
            
            for news in high_risk[:5]:
                html += self._create_email_news_item(news, '#dc3545')
            
            html += """
                        </td>
                    </tr>"""
        
        # MEDIUM RISK 뉴스
        if medium_risk:
            html += """
                    <tr>
                        <td style="padding: 0 30px 30px;">
                            <h2 style="color: #ffc107; font-size: 20px; margin-bottom: 15px;">📢 MEDIUM RISK - 주의 필요</h2>"""
            
            for news in medium_risk[:5]:
                html += self._create_email_news_item(news, '#ffc107')
            
            html += """
                        </td>
                    </tr>"""
        
        # LOW RISK 뉴스
        if low_risk:
            html += """
                    <tr>
                        <td style="padding: 0 30px 30px;">
                            <h2 style="color: #28a745; font-size: 20px; margin-bottom: 15px;">ℹ️ LOW RISK - 모니터링</h2>"""
            
            for news in low_risk[:5]:
                html += self._create_email_news_item(news, '#28a745')
            
            html += """
                        </td>
                    </tr>"""
        
        # 푸터
        html += """
                    <tr>
                        <td style="background-color: #f8f9fa; padding: 20px; text-align: center; border-radius: 0 0 8px 8px;">
                            <p style="color: #666; margin: 0; font-size: 12px;">
                                Samsung C&T Global Risk Monitoring System<br>
                                Powered by SerpAPI & Gemini AI<br>
                                Generated at """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """
                            </p>
                        </td>
                    </tr>
                    
                </table>
                
            </td>
        </tr>
    </table>
    
</body>
</html>"""
        
        return html
    
    def _create_email_news_item(self, news: NewsItem, color: str) -> str:
        """이메일용 뉴스 아이템 생성"""
        import html
        
        return f"""
        <table width="100%" cellpadding="0" cellspacing="0" border="0" style="margin-bottom: 20px; border-left: 4px solid {color}; background-color: #f9f9f9;">
            <tr>
                <td style="padding: 15px;">
                    <h3 style="margin: 0 0 10px 0; color: #333; font-size: 16px;">
                        {html.escape(news.title[:100])}...
                    </h3>
                    <table cellpadding="0" cellspacing="0" border="0">
                        <tr>
                            <td style="padding-right: 15px; color: #666; font-size: 12px;">
                                📍 {news.country_ko or news.country}
                            </td>
                            <td style="padding-right: 15px; color: #666; font-size: 12px;">
                                📰 {html.escape(news.source)}
                            </td>
                            <td style="color: #666; font-size: 12px;">
                                📅 {news.date[:10] if len(news.date) > 10 else news.date}
                            </td>
                        </tr>
                    </table>
                    <div style="margin: 10px 0; padding: 10px; background-color: #fff; border-radius: 4px;">
                        <strong style="color: #666; font-size: 12px;">리스크 점수:</strong> 
                        <span style="color: {color}; font-weight: bold;">{news.risk_score:.0f}점</span> | 
                        <strong style="color: #666; font-size: 12px;">카테고리:</strong> {news.risk_category or 'Other'}
                    </div>
                    <div style="margin: 10px 0; padding: 10px; background-color: #fffbf0; border-radius: 4px;">
                        <strong style="color: #666; font-size: 12px;">AI 요약:</strong><br>
                        <p style="margin: 5px 0 0 0; color: #333; font-size: 13px; line-height: 1.5;">
                            {html.escape(news.ai_summary_ko[:200] if news.ai_summary_ko else 'No summary available')}...
                        </p>
                    </div>
                    <a href="{news.link}" style="display: inline-block; margin-top: 10px; padding: 8px 15px; background-color: {color}; color: white; text-decoration: none; border-radius: 4px; font-size: 12px;">
                        원문 보기 →
                    </a>
                </td>
            </tr>
        </table>"""
    
    def run_daily_monitoring(self):
        """일일 전체 모니터링 (아침 7시)"""
        logger.info("\n" + "="*70)
        logger.info("🌅 일일 전체 리스크 모니터링 시작")
        logger.info("="*70)
        
        try:
            # 일일 캐시 초기화
            self.news_cache.clear_daily_cache()
            
            # 1. 국가별 뉴스 수집
            country_news = self.collect_country_news()
            
            # 2. 회사 관련 뉴스 수집
            company_news = self.collect_company_news()
            
            # 3. 통합
            all_news = country_news + company_news
            logger.info(f"✅ 총 {len(all_news)}건 수집 완료")
            
            # 4. 날짜 필터링
            days_to_keep = self.config.get('search_settings', {}).get('days_to_keep', 7)
            all_news = self.filter_recent_news(all_news, days=days_to_keep)
            
            # 5. AI 분석 진행
            unique_news = self.analyzer.remove_duplicates(all_news)
            analyzed_news = self.analyzer.analyze_risk_batch(unique_news)
            final_news = self.analyzer.summarize_and_translate(analyzed_news)
            
            # 6. 리포트 생성 및 전송
            html_content = self.create_ai_html_report(final_news)
            email_html = self.create_email_compatible_html_report(final_news)
            
            # 파일 저장
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            html_file = f'daily_risk_report_{timestamp}.html'
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # 이메일 전송
            if self.email_config['sender_email'] and self.email_config['recipients']:
                self.send_email_report(email_html, final_news, use_email_version=True)
            
            logger.info("✅ 일일 모니터링 완료")
            return True
            
        except Exception as e:
            logger.error(f"❌ 일일 모니터링 오류: {e}")
            return False
    
    def run_company_monitoring(self):
        """회사 관련 뉴스만 모니터링 (3시간마다)"""
        logger.info("\n" + "="*70)
        logger.info("🏢 회사 관련 뉴스 모니터링 시작")
        logger.info("="*70)
        
        try:
            # 1. 회사 뉴스만 수집
            company_news = self.collect_company_news()
            
            # 2. 날짜 필터링
            company_news = self.filter_recent_news(company_news, days=1)  # 최근 1일
            
            # 3. 새로운 뉴스만 필터링
            new_news = self.filter_new_company_news(company_news)
            
            if not new_news:
                logger.info("ℹ️ 새로운 회사 관련 뉴스 없음")
                return True
            
            # 4. AI 분석
            logger.info(f"🤖 {len(new_news)}건의 새로운 뉴스 분석 시작")
            unique_news = self.analyzer.remove_duplicates(new_news)
            analyzed_news = self.analyzer.analyze_risk_batch(unique_news)
            
            # 리스크가 있는 뉴스만 필터링
            risk_news = [n for n in analyzed_news if n.risk_level in ['HIGH', 'MEDIUM']]
            
            if risk_news:
                # 요약 및 번역
                final_news = self.analyzer.summarize_and_translate(risk_news)
                
                # 긴급 알림 이메일 생성
                email_html = self.create_urgent_email_report(final_news)
                
                # 이메일 전송
                if self.email_config['sender_email'] and self.email_config['recipients']:
                    subject = f"[긴급] 삼성물산 관련 리스크 발견 - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                    self.send_urgent_email(email_html, subject)
                
                logger.info(f"⚠️ {len(risk_news)}건의 리스크 뉴스 발견 및 알림 전송")
            else:
                logger.info("✅ 리스크 수준이 낮은 뉴스만 발견됨")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 회사 모니터링 오류: {e}")
            return False
    
    def create_urgent_email_report(self, news_list: List[NewsItem]) -> str:
        """긴급 알림용 간단한 이메일 HTML 생성"""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>삼성물산 관련 긴급 리스크 알림</title>
</head>
<body style="margin: 0; padding: 20px; font-family: Arial, sans-serif; background-color: #f4f4f4;">
    <div style="max-width: 600px; margin: 0 auto; background-color: #ffffff; border-radius: 8px; overflow: hidden;">
        <div style="background-color: #dc3545; padding: 20px; text-align: center;">
            <h1 style="color: #ffffff; margin: 0; font-size: 24px;">⚠️ 삼성물산 관련 리스크 감지</h1>
            <p style="color: #ffffff; margin: 10px 0 0 0; font-size: 14px;">
                {datetime.now().strftime('%Y년 %m월 %d일 %H:%M')} 수집
            </p>
        </div>
        
        <div style="padding: 20px;">
            <p style="color: #333; margin-bottom: 20px;">
                삼성물산 관련 새로운 리스크 뉴스 {len(news_list)}건이 발견되었습니다.
            </p>"""
        
        for news in news_list:
            color = '#dc3545' if news.risk_level == 'HIGH' else '#ffc107'
            html += f"""
            <div style="border-left: 4px solid {color}; padding: 15px; margin-bottom: 15px; background-color: #f9f9f9;">
                <h3 style="margin: 0 0 10px 0; color: #333; font-size: 16px;">
                    {news.title[:100]}...
                </h3>
                <p style="margin: 5px 0; color: #666; font-size: 12px;">
                    📰 {news.source} | 📅 {news.date[:10] if len(news.date) > 10 else news.date}
                </p>
                <p style="margin: 5px 0; color: {color}; font-weight: bold; font-size: 14px;">
                    리스크: {news.risk_level} ({news.risk_score:.0f}점)
                </p>
                <p style="margin: 10px 0; color: #333; font-size: 13px;">
                    {news.ai_summary_ko[:150] if news.ai_summary_ko else 'No summary'}...
                </p>
                <a href="{news.link}" style="color: #007bff; text-decoration: none; font-size: 12px;">
                    원문 보기 →
                </a>
            </div>"""
        
        html += """
        </div>
    </div>
</body>
</html>"""
        return html
    
    def send_urgent_email(self, html_content: str, subject: str) -> bool:
        """긴급 이메일 전송"""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_config['sender_email']
            msg['To'] = ', '.join(self.email_config['recipients'])
            
            html_part = MIMEText(html_content, 'html', 'utf-8')
            msg.attach(html_part)
            
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['sender_email'], self.email_config['sender_password'])
                server.send_message(msg)
                
            logger.info(f"📧 긴급 이메일 전송 성공")
            return True
            
        except Exception as e:
            logger.error(f"이메일 전송 실패: {e}")
            return False
    
    def run(self):
        """테스트용 기본 실행 메서드 (부모 클래스 오버라이드)"""
        logger.info("\n" + "="*70)
        logger.info("🧪 테스트 모드 실행")
        logger.info("="*70)
        
        # 전체 모니터링 실행 (일일 모니터링과 동일)
        return self.run_daily_monitoring()
    
    def start_scheduler(self):
        """스케줄러 시작"""
        logger.info("⏰ 스케줄러 시작")
        
        # 매일 아침 7시 전체 모니터링
        schedule.every().day.at("07:00").do(self.run_daily_monitoring)
        
        # 3시간마다 회사 모니터링
        schedule.every(3).hours.do(self.run_company_monitoring)
        
        logger.info("📅 스케줄 설정 완료:")
        logger.info("  - 일일 전체 모니터링: 매일 07:00")
        logger.info("  - 회사 관련 모니터링: 3시간마다")
        
        # 스케줄러 실행
        while True:
            schedule.run_pending()
            time.sleep(60)  # 1분마다 체크


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='AI 기반 글로벌 리스크 모니터링')
    parser.add_argument('--mode', choices=['normal', 'test', 'schedule', 'company'], 
                       default='test',
                       help='실행 모드: normal(일반), test(테스트), schedule(스케줄링), company(회사만)')
    parser.add_argument('--config', default='monitoring_config.json', 
                       help='설정 파일 경로')
    args = parser.parse_args()
    
    try:
        monitor = EnhancedAIRiskMonitoringSystem(args.config, mode=args.mode)
        
        if args.mode == 'test':
            # 테스트 모드: 1회 실행
            logger.info("🧪 테스트 모드 - 1회 실행")
            monitor.run()
            
        elif args.mode == 'normal':
            # 일반 모드: 일일 전체 모니터링 1회 실행
            logger.info("📊 일반 모드 - 전체 모니터링 실행")
            monitor.run_daily_monitoring()
            
        elif args.mode == 'company':
            # 회사 모드: 회사 관련만 1회 실행
            logger.info("🏢 회사 모드 - 회사 관련 모니터링 실행")
            monitor.run_company_monitoring()
            
        elif args.mode == 'schedule':
            # 스케줄 모드: 지속적 실행
            logger.info("⏰ 스케줄 모드 - 자동 스케줄링 시작")
            monitor.start_scheduler()
            
    except KeyboardInterrupt:
        logger.info("\n사용자에 의해 중단되었습니다")
    except Exception as e:
        logger.error(f"치명적 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()
            