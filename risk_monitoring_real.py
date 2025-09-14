"""
AI 기반 글로벌 리스크 모니터링 시스템 - 스케줄링 및 회사 전용 모니터링 추가
24개국 실시간 뉴스 모니터링 with AI 분석
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

class CompanyNewsCache:
    """회사 뉴스 캐시 관리 (3시간 주기 체크용)"""
    
    def __init__(self, cache_file='company_news_cache.pkl'):
        self.cache_file = cache_file
        self.cache = self.load_cache()
    
    def load_cache(self) -> Set[str]:
        """캐시 로드"""
        if Path(self.cache_file).exists():
            try:
                with open(self.cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                return set()
        return set()
    
    def save_cache(self):
        """캐시 저장"""
        with open(self.cache_file, 'wb') as f:
            pickle.dump(self.cache, f)
    
    def is_new_news(self, news_hash: str) -> bool:
        """새로운 뉴스인지 확인"""
        return news_hash not in self.cache
    
    def add_news(self, news_hash: str):
        """뉴스 해시 추가"""
        self.cache.add(news_hash)
    
    def clear_old_cache(self):
        """오래된 캐시 정리 (선택적)"""
        pass

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
    news_hash: str = ""
    
    # AI analysis result fields
    risk_score: float = 0.0
    risk_level: str = ""
    risk_category: str = ""
    ai_summary_ko: str = ""
    ai_title_ko: str = ""  # 한국어 제목 추가
    ai_full_translation_ko: str = ""
    is_duplicate: bool = False
    duplicate_of: str = ""
    ai_analysis_timestamp: str = ""
    
    def __post_init__(self):
        """뉴스 해시 생성"""
        if not self.news_hash:
            content = f"{self.title}{self.snippet}{self.source}"
            self.news_hash = hashlib.md5(content.encode()).hexdigest()

class GeminiAnalyzer:
    """Gemini AI 분석 클래스"""
    
    def __init__(self, api_key: str):
        """Gemini API 초기화"""
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        logger.info("✅ Gemini 2.0 Flash 초기화 완료")
        
        # 리스크 점수 기준
        self.risk_thresholds = {
            'HIGH': 70,
            'MEDIUM': 40,
            'LOW': 20
        }
    
    def remove_duplicates(self, news_list: List[NewsItem]) -> List[NewsItem]:
        """AI 기반 중복 뉴스 제거"""
        logger.info("🔍 AI 기반 중복 뉴스 제거 시작...")
        
        if not news_list:
            return []
        
        unique_news = []
        duplicate_count = 0
        
        country_news = {}
        for news in news_list:
            country = news.country
            if country not in country_news:
                country_news[country] = []
            country_news[country].append(news)
        
        for country, items in country_news.items():
            if not items:
                continue
            
            items.sort(key=lambda x: x.date, reverse=True)
            unique_news.append(items[0])
            
            for i in range(1, len(items)):
                candidate = items[i]
                is_duplicate = False
                duplicate_of = None
                
                country_unique = [n for n in unique_news if n.country == country]
                
                if country_unique:
                    is_duplicate, duplicate_of = self._check_duplicate_with_ai(
                        candidate, 
                        country_unique[-min(10, len(country_unique)):]
                    )
                
                if not is_duplicate:
                    unique_news.append(candidate)
                else:
                    duplicate_count += 1
                    candidate.is_duplicate = True
                    candidate.duplicate_of = duplicate_of or ""
                    logger.debug(f"  - 중복 제거: {candidate.title[:50]}...")
            
            original_count = len(items)
            final_count = len([n for n in unique_news if n.country == country])
            logger.info(f"  - {country}: {original_count}건 → {final_count}건")
        
        logger.info(f"✅ AI 중복 제거 완료: {len(news_list)}건 → {len(unique_news)}건")
        return unique_news
    
    def _check_duplicate_with_ai(self, candidate: NewsItem, existing_news: List[NewsItem]) -> Tuple[bool, Optional[str]]:
        """AI를 사용한 중복 체크"""
        try:
            prompt = f"""You are a news deduplication expert. Analyze if the candidate news is reporting the EXACT SAME INCIDENT as any existing news.

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

RESPONSE FORMAT:
IsDuplicate: (Yes/No)
DuplicateNumber: (1-10, or 0 if not duplicate)
Reason: (Brief explanation)"""
            
            response = self.model.generate_content(prompt)
            result = response.text.strip()
            
            is_duplicate = False
            duplicate_idx = 0
            
            lines = result.split('\n')
            for line in lines:
                line_lower = line.lower()
                if 'isduplicate:' in line_lower:
                    is_duplicate = 'yes' in line_lower and 'no' not in line_lower
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
        
        filtered_list = news_list
        analyzed_news = []
        
        # 배치 처리
        for i in range(0, len(filtered_list), batch_size):
            batch = filtered_list[i:i+batch_size]
            
            # 회사 뉴스는 AI 분석 없이 바로 COMPANY로 분류
            company_batch = []
            regular_batch = []
            
            for news in batch:
                if news.country_code in ["samsung", "global_samsung"]:
                    news.risk_level = 'COMPANY'
                    news.risk_score = 0
                    news.risk_category = 'Company News'
                    news.ai_analysis_timestamp = datetime.now().isoformat()
                    company_batch.append(news)
                else:
                    regular_batch.append(news)
            
            # 일반 뉴스만 AI 분석
            if regular_batch:
                prompt = self._create_risk_analysis_prompt(regular_batch)
                
                try:
                    response = self.model.generate_content(prompt)
                    results = self._parse_risk_response(response.text, regular_batch)
                    analyzed_news.extend(results)
                    
                    time.sleep(1)
                    logger.info(f"  - 분석 진행: {min(i+batch_size, len(filtered_list))}/{len(filtered_list)}")
                    
                except Exception as e:
                    logger.error(f"❌ AI 분석 오류: {e}")
                    for news in regular_batch:
                        news.risk_score = 0
                        news.risk_level = ""
                    analyzed_news.extend(regular_batch)
            
            # 회사 뉴스 추가
            analyzed_news.extend(company_batch)
        
        # 리스크 점수 기준으로 필터링
        filtered_news = []
        for n in analyzed_news:
            # 회사 뉴스는 무조건 포함
            if n.risk_level == 'COMPANY':
                filtered_news.append(n)
            # 일반 뉴스는 LOW(20점) 이상만 포함
            elif n.risk_score >= self.risk_thresholds['LOW']:
                filtered_news.append(n)
        
        logger.info(f"✅ AI 분석 완료: {len(filtered_news)}건 처리")
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

Note: If news source is from Samsung official channels, deduct points appropriately.

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
        sections = response_text.split('[')
        
        for section in sections[1:]:
            try:
                lines = section.strip().split('\n')
                news_idx = int(lines[0].split(']')[0]) - 1
                
                if news_idx >= len(news_batch):
                    continue
                
                news = news_batch[news_idx]
                
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
                
                # 리스크 레벨 설정
                if news.risk_score >= self.risk_thresholds['HIGH']:
                    news.risk_level = 'HIGH'
                elif news.risk_score >= self.risk_thresholds['MEDIUM']:
                    news.risk_level = 'MEDIUM'
                else:
                    news.risk_level = 'LOW'
                
                news.ai_analysis_timestamp = datetime.now().isoformat()
                results.append(news)
                
            except Exception as e:
                logger.error(f"파싱 오류: {e}")
                continue
        
        return results
    
    def summarize_and_translate(self, news_list: List[NewsItem]) -> List[NewsItem]:
        """뉴스 요약 및 한국어 번역"""
        logger.info("📝 뉴스 요약 및 번역 시작...")
        
        # 리스크 레벨이 있는 뉴스만 카운트 (COMPANY 포함)
        total_items = len([n for n in news_list if n.risk_level in ['HIGH', 'MEDIUM', 'LOW', 'COMPANY']])
        processed = 0
        
        for news in news_list:
            if not news.risk_level:
                continue
            
            try:
                # HIGH는 전체 번역, 나머지는 요약만
                if news.risk_level == 'HIGH':
                    prompt = f"""Please translate the title and content, then summarize the following news into Korean.

    Title: {news.title}
    Content: {news.snippet}
    Date: {news.date}
    Country: {news.country}

    Please respond in the following format:
    [Title Translation]
    (Korean translation of the title)

    [Summary]
    (3-4 sentences summarizing key points in Korean)

    [Full Translation]
    (Complete translation of the content in natural Korean)"""
                    
                else:  # MEDIUM, LOW, COMPANY
                    prompt = f"""Please translate the title and summarize the following news in 3-4 sentences in Korean.

    Title: {news.title}
    Content: {news.snippet}
    Date: {news.date}
    Country: {news.country}

    Please respond in the following format:
    [Title Translation]
    (Korean translation of the title)

    [Summary]
    (3-4 sentences summarizing key points in Korean)"""
                
                response = self.model.generate_content(prompt)
                result = response.text
                
                # 결과 파싱
                if '[Title Translation]' in result:
                    if news.risk_level == 'HIGH':
                        parts = result.split('[Title Translation]')[1]
                        if '[Summary]' in parts:
                            title_ko = parts.split('[Summary]')[0].strip()
                            remaining = parts.split('[Summary]')[1]
                            if '[Full Translation]' in remaining:
                                summary = remaining.split('[Full Translation]')[0].strip()
                                translation = remaining.split('[Full Translation]')[1].strip()
                            else:
                                summary = remaining.strip()
                                translation = ""
                        else:
                            title_ko = parts.strip()
                            summary = ""
                            translation = ""
                        
                        news.ai_title_ko = title_ko
                        news.ai_summary_ko = summary
                        news.ai_full_translation_ko = translation
                        
                    else:  # MEDIUM, LOW, COMPANY
                        if '[Summary]' in result:
                            title_ko = result.split('[Title Translation]')[1].split('[Summary]')[0].strip()
                            summary = result.split('[Summary]')[1].strip()
                            news.ai_title_ko = title_ko
                            news.ai_summary_ko = summary
                        else:
                            news.ai_title_ko = news.title
                            news.ai_summary_ko = result.strip()
                
                time.sleep(0.5)
                processed += 1
                
                if processed % 10 == 0:
                    logger.info(f"  - 번역 진행: {processed}/{total_items}")
                    
            except Exception as e:
                logger.error(f"번역/요약 오류 ({news.title[:50]}...): {e}")
                news.ai_title_ko = news.title
                news.ai_summary_ko = "번역 실패"
        
        logger.info(f"✅ 요약 및 번역 완료: {processed}건 처리")
        return news_list

class AIRiskMonitoringSystem:
    """AI 기반 리스크 모니터링 시스템"""
    
    def __init__(self, config_path='monitoring_config.json'):
        """시스템 초기화"""
        logger.info("="*70)
        logger.info("🤖 AI 기반 글로벌 리스크 모니터링 시스템 초기화")
        logger.info("="*70)
        
        self.serpapi_key = os.getenv('SERPAPI_KEY')
        self.gemini_key = os.getenv('GEMINI_API_KEY')
        
        if not self.serpapi_key:
            logger.error("❌ SERPAPI_KEY가 설정되지 않았습니다.")
            sys.exit(1)
        
        if not self.gemini_key:
            logger.error("❌ GEMINI_API_KEY가 설정되지 않았습니다.")
            sys.exit(1)
        
        try:
            from serpapi import GoogleSearch
            self.GoogleSearch = GoogleSearch
            logger.info("✅ SerpAPI 패키지 로드 완료")
        except ImportError:
            logger.error("❌ serpapi 패키지가 설치되지 않았습니다.")
            sys.exit(1)
        
        self.analyzer = GeminiAnalyzer(self.gemini_key)
        self.load_config(config_path)
        self.setup_email_config()
        
        # 수정된 통계 정보 - 완전한 통계
        self.stats = {
            'api_calls': 0,
            'news_collected': 0,
            'news_after_dedup': 0,
            'news_analyzed': 0,
            'high_risk': 0,
            'medium_risk': 0,
            'low_risk': 0,  # 추가
            'company_news': 0,  # 추가
            'total_filtered': 0,  # 추가
            'errors': 0,
            'start_time': datetime.now(),
            'country_breakdown': {}  # 추가 - 국가별 통계
        }

    def update_risk_statistics(self, news_list: List[NewsItem]):
        """리스크 통계 업데이트 - 새로운 메서드"""
        self.stats['high_risk'] = 0
        self.stats['medium_risk'] = 0
        self.stats['low_risk'] = 0
        self.stats['company_news'] = 0
        self.stats['country_breakdown'] = {}
        
        for news in news_list:
            # 리스크 레벨별 카운트
            if news.risk_level == 'HIGH':
                self.stats['high_risk'] += 1
            elif news.risk_level == 'MEDIUM':
                self.stats['medium_risk'] += 1
            elif news.risk_level == 'LOW':
                self.stats['low_risk'] += 1
            elif news.risk_level == 'COMPANY':
                self.stats['company_news'] += 1
            
            # 국가별 통계
            country = news.country_ko or news.country
            if country not in self.stats['country_breakdown']:
                self.stats['country_breakdown'][country] = {
                    'total': 0, 'high': 0, 'medium': 0, 'low': 0, 'company': 0
                }
            
            self.stats['country_breakdown'][country]['total'] += 1
            if news.risk_level == 'HIGH':
                self.stats['country_breakdown'][country]['high'] += 1
            elif news.risk_level == 'MEDIUM':
                self.stats['country_breakdown'][country]['medium'] += 1
            elif news.risk_level == 'LOW':
                self.stats['country_breakdown'][country]['low'] += 1
            elif news.risk_level == 'COMPANY':
                self.stats['country_breakdown'][country]['company'] += 1
        
        self.stats['total_filtered'] = len(news_list)
 
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
            'recipients': [],
            'admin_email': os.getenv('ADMIN_EMAIL', '')  # 관리자 이메일 추가
        }
        
        env_recipients = os.getenv('RECIPIENT_EMAILS', '')
        if env_recipients:
            self.email_config['recipients'] = [
                email.strip() for email in env_recipients.split(',')
            ]

    def search_news(self, query: str, country_code: str = None, 
                country_name: str = None, search_type: str = 'news') -> List[NewsItem]:
        """뉴스 검색 - 7일 이내 뉴스만"""
        results = []
        
        # 현재 시간과 7일 전 시간 설정
        now = datetime.now()
        seven_days_ago = now - timedelta(days=7)
        
        logger.debug(f"날짜 필터: {seven_days_ago.strftime('%Y-%m-%d')} ~ {now.strftime('%Y-%m-%d')}")
        
        try:
            if search_type == 'news':
                params = {
                    "api_key": self.serpapi_key,
                    "engine": "google_news",
                    "q": query,
                    "when": "7d"  # 최근 7일
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
                    "tbs": "qdr:w"  # 최근 1주일
                }
                
                if country_code:
                    params["gl"] = country_code
                    params["hl"] = "ko" if country_code == "kr" else "en"
            
            search = self.GoogleSearch(params)
            response = search.get_dict()
            
            self.stats['api_calls'] += 1
            
            # 결과 파싱
            if search_type == 'news' and "news_results" in response:
                for item in response["news_results"][:20]:
                    date_str = item.get('date', '')
                    
                    # 통일된 날짜 검증 함수 사용 (수정된 부분)
                    if not self._is_within_days(date_str, 7):
                        logger.debug(f"  ✗ 7일 이전 뉴스 제외: {item.get('title', '')[:50]}... ({date_str})")
                        continue
                    
                    logger.debug(f"  ✔ 포함: {item.get('title', '')[:30]}... ({date_str})")
                    
                    news_item = NewsItem(
                        title=item.get('title', ''),
                        date=date_str,
                        source=item.get('source', {}).get('name', 'Unknown'),
                        snippet=item.get('snippet', ''),
                        link=item.get('link', ''),
                        country=country_name or 'Global',
                        country_code=country_code or 'global',
                        thumbnail=item.get('thumbnail', ''),
                        search_type=search_type,
                        collected_at=now.isoformat()
                    )
                    results.append(news_item)
                    
            elif search_type != 'news' and "organic_results" in response:
                # Google Search는 이미 최신 결과만 반환
                for item in response["organic_results"][:20]:
                    news_item = NewsItem(
                        title=item.get('title', ''),
                        date=now.strftime('%Y-%m-%d'),
                        source=item.get('displayed_link', 'Unknown'),
                        snippet=item.get('snippet', ''),
                        link=item.get('link', ''),
                        country=country_name or 'Korea',
                        country_code=country_code or 'kr',
                        search_type=search_type,
                        collected_at=now.isoformat()
                    )
                    results.append(news_item)
            
            self.stats['news_collected'] += len(results)
            
            if results:
                logger.info(f"  ✔ {len(results)}건 수집 (7일 이내)")
            else:
                logger.info(f"  ✗ 7일 이내 뉴스 없음")
            
        except Exception as e:
            logger.error(f"API 오류: {e}")
            self.stats['errors'] += 1
        
        return results
    
    def collect_all_news(self) -> List[NewsItem]:
        """모든 뉴스 수집"""
        all_news = []
        
        # 1. 국가별 리스크 뉴스 수집
        logger.info("\n🌍 국가별 리스크 뉴스 수집 시작")
        for idx, (country_code, country_info) in enumerate(self.countries.items(), 1):
            logger.info(f"[{idx}/{len(self.countries)}] {country_info['name_ko']} ({country_info['name']})")
            
            query = f'("{country_info["name"]}" disaster OR "{country_info["name"]}" accident OR "{country_info["name"]}" crisis OR "{country_info["name"]}" emergency)'
            
            news = self.search_news(
                query=query,
                country_code=country_info['gl'],
                country_name=country_info['name'],
                search_type='news'
            )
            
            for item in news:
                item.country_ko = country_info['name_ko']
            
            logger.info(f"  - {country_info['name']}: {len(news)}건 수집")
            all_news.extend(news)
            time.sleep(1)
        
        # 2. 회사 키워드 뉴스 수집 (해외만) - 수정됨
        logger.info("\n🏢 회사 키워드 뉴스 수집 시작 (해외만)")
        
        # 제외할 한국 언론사
        korean_sources = ['yonhap', '연합', 'korea', 'chosun', '조선', 
                        'joongang', '중앙', 'hankyoreh', '한겨레', 'donga', '동아',
                        'hankook', '한국', 'maeil', '매일', 'seoul', '서울']
        
        # 제외할 회사 공식 채널
        official_sources = ['samsung newsroom', '삼성 뉴스룸', 'samsung.com', 
                        'samsungcnt.com', 'samsung c&t newsroom']
        
        # 건설업 관련 키워드 (필터링용)
        construction_keywords = ['construction', 'building', 'infrastructure', 'engineering',
                                'project', 'development', 'contractor', 'architecture',
                                '건설', '건축', '공사', '시공', '프로젝트', '개발']
        
        for idx, keyword in enumerate(self.company_keywords, 1):
            logger.info(f"[{idx}/{len(self.company_keywords)}] {keyword}")
            
            # 건설업 관련 키워드 포함한 검색어
            query = f'"{keyword}" (construction OR building OR project OR infrastructure) -site:kr -korea -한국 -newsroom'
            
            try:
                params = {
                    "api_key": self.serpapi_key,
                    "engine": "google_news",
                    "q": query,
                    "when": "7d",
                    "gl": "us",
                    "hl": "en"
                }
                
                search = self.GoogleSearch(params)
                response = search.get_dict()
                
                self.stats['api_calls'] += 1
                
                if "news_results" in response:
                    company_news = []
                    
                    for item in response["news_results"][:30]:  # 더 많이 가져와서 필터링
                        # 날짜 필터링
                        date_str = item.get('date', '')
                        if not self._is_within_days(date_str, 7):
                            continue
                        
                        # 소스 필터링
                        source = item.get('source', {}).get('name', '').lower()
                        
                        # 한국 언론사 제외
                        if any(ks in source for ks in korean_sources):
                            logger.debug(f"  ✗ 한국 언론사 제외: {source}")
                            continue
                        
                        # 회사 공식 채널 제외
                        if any(os in source for os in official_sources):
                            logger.debug(f"  ✗ 회사 공식 채널 제외: {source}")
                            continue
                        
                        # 제목과 내용에서 건설업 관련성 체크
                        title = item.get('title', '').lower()
                        snippet = item.get('snippet', '').lower()
                        
                        # 건설업 관련 키워드가 하나도 없으면 제외
                        has_construction_relevance = any(
                            ck.lower() in title or ck.lower() in snippet 
                            for ck in construction_keywords
                        )
                        
                        if not has_construction_relevance:
                            logger.debug(f"  ✗ 건설업 무관: {item.get('title', '')[:50]}...")
                            continue
                        
                        news_item = NewsItem(
                            title=item.get('title', ''),
                            date=date_str,
                            source=item.get('source', {}).get('name', 'Unknown'),
                            snippet=item.get('snippet', ''),
                            link=item.get('link', ''),
                            country="Global",
                            country_ko="해외",
                            country_code="global_samsung",
                            thumbnail=item.get('thumbnail', ''),
                            search_type='company_global',
                            collected_at=datetime.now().isoformat()
                        )
                        company_news.append(news_item)
                        
                        # 최대 10건만 수집
                        if len(company_news) >= 10:
                            break
                    
                    logger.info(f"  - {keyword}: {len(company_news)}건 수집 (건설업 관련, 공식채널 제외)")
                    all_news.extend(company_news)
                    self.stats['news_collected'] += len(company_news)
                    
            except Exception as e:
                logger.error(f"회사 키워드 검색 오류 ({keyword}): {e}")
                self.stats['errors'] += 1
            
            time.sleep(1)
        
        # 3. 한국 미디어 검색 추가 (기존 유지)
        logger.info("\n🇰🇷 한국 언론 내 회사 뉴스 모니터링")
        for site in self.korean_media.get('sites', []):
            if not site.get('active', False):
                continue
            
            for term in self.korean_media.get('search_terms', []):
                query = f'{site["selector"]} "{term}"'
                logger.info(f"  - {site['name']}: {term}")
                news = self.search_news(
                    query=query,
                    country_code='kr',
                    country_name='Korea',
                    search_type='web'
                )
                
                for item in news:
                    item.country = "Samsung C&T"
                    item.country_ko = "삼성물산"
                    item.country_code = "samsung"
                
                all_news.extend(news)
                time.sleep(1)
        
        logger.info(f"\n✅ 총 {len(all_news)}건 수집 완료")
        return all_news
    
    def _is_within_days(self, date_string: str, days: int = 7) -> bool:
        """통일된 날짜 검증 함수 - 지정된 일수 이내인지 확인"""
        try:
            target_date = datetime.now() - timedelta(days=days)
            date_lower = date_string.lower()
            
            # 상대적 시간 처리
            relative_terms = ['today', 'yesterday', 'hour ago', 'hours ago', 
                            'minute ago', 'minutes ago', 'just now']
            if any(term in date_lower for term in relative_terms):
                return True
            
            # X days ago 패턴
            if 'day ago' in date_lower or 'days ago' in date_lower:
                days_match = re.search(r'(\d+)\s*days?\s*ago', date_lower)
                if days_match:
                    days_num = int(days_match.group(1))
                    return days_num <= days
            
            # 절대 날짜 파싱 - 다양한 형식 시도
            date_formats = ['%m/%d/%Y', '%Y-%m-%d', '%d %B %Y', '%B %d, %Y', 
                        '%d/%m/%Y', '%Y/%m/%d']
            
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(date_string.split(',')[0].strip(), fmt)
                    return parsed_date >= target_date
                except:
                    continue
            
            # dateutil parser 사용 (최후의 수단)
            try:
                parsed_date = parser.parse(date_string, fuzzy=True)
                # 미래 날짜면 작년으로 조정
                if parsed_date > datetime.now():
                    parsed_date = parsed_date.replace(year=parsed_date.year - 1)
                return parsed_date >= target_date
            except:
                pass
            
            # 월 이름 패턴 특별 처리
            months_pattern = r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2})'
            month_match = re.search(months_pattern, date_string)
            if month_match:
                month_str, day = month_match.groups()
                current_year = datetime.now().year
                try:
                    test_date = parser.parse(f"{month_str} {day}, {current_year}")
                    if test_date > datetime.now():
                        test_date = parser.parse(f"{month_str} {day}, {current_year - 1}")
                    return test_date >= target_date
                except:
                    pass
            
            # 파싱 실패 시 제외 (안전한 선택)
            logger.debug(f"날짜 파싱 실패: {date_string}")
            return False
            
        except Exception as e:
            logger.debug(f"날짜 검증 오류: {date_string} - {e}")
            return False
    
    def create_ai_html_report(self, analyzed_news: List[NewsItem]) -> str:
        """AI 분석 결과를 포함한 HTML 리포트 생성 - HIGH, MEDIUM, LOW, COMPANY 모두 포함"""
        # HIGH, MEDIUM, LOW, COMPANY 분류
        high_risk = [n for n in analyzed_news if n.risk_level == 'HIGH']
        medium_risk = [n for n in analyzed_news if n.risk_level == 'MEDIUM']
        low_risk = [n for n in analyzed_news if n.risk_level == 'LOW']
        company_news = [n for n in analyzed_news if n.risk_level == 'COMPANY']
        
        # 국가별 통계 계산
        country_stats = {}
        for news in analyzed_news:
            country = news.country_ko or news.country
            if country not in country_stats:
                country_stats[country] = {'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'COMPANY': 0, 'total': 0}
            
            if news.risk_level in ['HIGH', 'MEDIUM', 'LOW', 'COMPANY']:
                country_stats[country][news.risk_level] += 1
                country_stats[country]['total'] += 1
        
        # 인라인 스타일로 통일된 HTML
        html = f"""<!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>글로벌 리스크 모니터링 리포트 - {datetime.now().strftime('%Y-%m-%d')}</title>
    </head>
    <body style="margin: 0; padding: 0; font-family: 'Malgun Gothic', Arial, sans-serif; background-color: #f4f4f4;">
        <div style="max-width: 800px; margin: 0 auto; background-color: #ffffff;">
            
            <!-- 헤더 -->
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px; text-align: center;">
                <h1 style="color: #ffffff; margin: 0 0 10px 0; font-size: 32px;">🌍 G/O실 글로벌 리스크 모니터링 리포트</h1>
                <div style="display: inline-block; background: rgba(255,255,255,0.2); padding: 5px 15px; border-radius: 20px; color: white; font-size: 14px;">
                    Powered by Gemini 2.0 Flash
                </div>
            </div>
            
            <!-- 통계 카드 -->
            <div style="background: #f8f9fa; padding: 30px; border-bottom: 2px solid #e9ecef;">
                <table style="width: 100%; border-collapse: collapse;">
                    <tr>
                        <td style="text-align: center; padding: 15px;">
                            <div style="font-size: 36px; font-weight: bold;">{len(analyzed_news)}</div>
                            <div style="color: #666; margin-top: 5px; font-size: 12px;">Total News</div>
                        </td>
                        <td style="text-align: center; padding: 15px;">
                            <div style="font-size: 36px; font-weight: bold; color: #dc3545;">{len(high_risk)}</div>
                            <div style="color: #666; margin-top: 5px; font-size: 12px;">High Risk</div>
                        </td>
                        <td style="text-align: center; padding: 15px;">
                            <div style="font-size: 36px; font-weight: bold; color: #ffc107;">{len(medium_risk)}</div>
                            <div style="color: #666; margin-top: 5px; font-size: 12px;">Medium Risk</div>
                        </td>
                        <td style="text-align: center; padding: 15px;">
                            <div style="font-size: 36px; font-weight: bold; color: #28a745;">{len(low_risk)}</div>
                            <div style="color: #666; margin-top: 5px; font-size: 12px;">Low Risk</div>
                        </td>
                        <td style="text-align: center; padding: 15px;">
                            <div style="font-size: 36px; font-weight: bold; color: #6c757d;">{len(company_news)}</div>
                            <div style="color: #666; margin-top: 5px; font-size: 12px;">Company</div>
                        </td>
                    </tr>
                </table>
            </div>
            
            <!-- 국가별 리스크 현황 -->
            <div style="padding: 30px; background-color: #f8f9fa;">
                <h2 style="margin: 0 0 20px 0; color: #333;">📊 국가별 리스크 현황</h2>
                <table style="width: 100%; border-collapse: collapse; background: white; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                    <thead>
                        <tr style="background-color: #6c757d; color: white;">
                            <th style="padding: 12px; text-align: left; border: 1px solid #dee2e6;">국가</th>
                            <th style="padding: 12px; text-align: center; border: 1px solid #dee2e6;">HIGH</th>
                            <th style="padding: 12px; text-align: center; border: 1px solid #dee2e6;">MEDIUM</th>
                            <th style="padding: 12px; text-align: center; border: 1px solid #dee2e6;">LOW</th>
                            <th style="padding: 12px; text-align: center; border: 1px solid #dee2e6;">COMPANY</th>
                            <th style="padding: 12px; text-align: center; border: 1px solid #dee2e6;">소계</th>
                        </tr>
                    </thead>
                    <tbody>"""
        
        # 국가별 통계 정렬 및 표시
        for country, stats in sorted(country_stats.items(), key=lambda x: x[1]['total'], reverse=True):
            html += f"""
                        <tr>
                            <td style="padding: 10px; border: 1px solid #dee2e6;">{country}</td>
                            <td style="padding: 10px; text-align: center; border: 1px solid #dee2e6;">
                                {f'<span style="background: #dc3545; color: white; padding: 2px 8px; border-radius: 4px;">{stats["HIGH"]}</span>' if stats['HIGH'] > 0 else '-'}
                            </td>
                            <td style="padding: 10px; text-align: center; border: 1px solid #dee2e6;">
                                {f'<span style="background: #ffc107; color: black; padding: 2px 8px; border-radius: 4px;">{stats["MEDIUM"]}</span>' if stats['MEDIUM'] > 0 else '-'}
                            </td>
                            <td style="padding: 10px; text-align: center; border: 1px solid #dee2e6;">
                                {f'<span style="background: #28a745; color: white; padding: 2px 8px; border-radius: 4px;">{stats["LOW"]}</span>' if stats['LOW'] > 0 else '-'}
                            </td>
                            <td style="padding: 10px; text-align: center; border: 1px solid #dee2e6;">
                                {f'<span style="background: #6c757d; color: white; padding: 2px 8px; border-radius: 4px;">{stats["COMPANY"]}</span>' if stats['COMPANY'] > 0 else '-'}
                            </td>
                            <td style="padding: 10px; text-align: center; border: 1px solid #dee2e6; font-weight: bold;">{stats['total']}</td>
                        </tr>"""
        
        html += """
                    </tbody>
                </table>
            </div>
            
            <!-- 뉴스 내용 -->
            <div style="padding: 40px;">
    """
        
        # COMPANY NEWS 섹션 (최상단)
        if company_news:
            html += """
                <h2 style="color: #6c757d; margin: 30px 0 20px 0;">🏢 COMPANY NEWS - 삼성물산 관련</h2>"""
            for news in company_news:
                html += self._create_ai_news_card(news, 'company')
        
        # HIGH RISK 섹션
        if high_risk:
            html += """
                <h2 style="color: #dc3545; margin: 30px 0 20px 0;">⚠️ HIGH RISK - 즉시 확인 필요</h2>"""
            for news in high_risk:
                html += self._create_ai_news_card(news, 'high')
        
        # MEDIUM RISK 섹션
        if medium_risk:
            html += """
                <h2 style="color: #ffc107; margin: 30px 0 20px 0;">🔔 MEDIUM RISK - 주의 필요</h2>"""
            for news in medium_risk:
                html += self._create_ai_news_card(news, 'medium')
        
        # LOW RISK 섹션
        if low_risk:
            html += """
                <h2 style="color: #28a745; margin: 30px 0 20px 0;">ℹ️ LOW RISK - 모니터링</h2>"""
            for news in low_risk:
                html += self._create_ai_news_card(news, 'low')
        
        html += """
            </div>
            
            <!-- 푸터 -->
            <div style="background-color: #f8f9fa; padding: 20px; text-align: center; border-top: 1px solid #dee2e6;">
                <p style="margin: 0; color: #666; font-size: 12px;">
                    Samsung C&T Risk Monitoring System<br>
                    Generated at """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """
                </p>
            </div>
        </div>
    </body>
    </html>"""
        
        return html
    
    def _create_ai_news_card(self, news: NewsItem, risk_class: str) -> str:
        """뉴스 카드 HTML 생성"""
        import html
        
        # 한국어 제목 우선 사용
        title_to_display = news.ai_title_ko if news.ai_title_ko else news.title
        
        # 리스크 레벨에 따른 색상
        color_map = {
            'high': '#dc3545',
            'medium': '#ffc107', 
            'low': '#28a745',
            'company': '#6c757d'
        }
        border_color = color_map.get(risk_class, '#6c757d')
        
        # 리스크 점수 표시 - COMPANY는 점수 대신 카테고리만 표시
        if risk_class == 'company':
            risk_info = f"<strong>카테고리:</strong> 회사 관련 뉴스"
        else:
            risk_info = f"""<strong>리스크 점수:</strong> {news.risk_score:.0f} | 
                            <strong>카테고리:</strong> {news.risk_category or 'Other'}"""
        
        return f"""
        <div style="background: white; border: 1px solid #e9ecef; border-left: 5px solid {border_color}; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
            <h3 style="margin: 0 0 10px 0; color: #333; font-size: 18px;">{html.escape(title_to_display)}</h3>
            <p style="margin: 10px 0; color: #666; font-size: 13px;">
                📍 {news.country_ko or news.country} | 📰 {html.escape(news.source)} | 📅 {news.date}
            </p>
            <p style="margin: 10px 0;">
                {risk_info}
            </p>
            <div style="background: #f8f9fa; padding: 10px; border-radius: 4px; margin: 15px 0;">
                <strong>AI 요약:</strong><br>
                <p style="margin: 5px 0; color: #333; line-height: 1.6;">
                    {html.escape(news.ai_summary_ko or 'No summary')}
                </p>
            </div>
            <a href="{news.link}" target="_blank" style="display: inline-block; padding: 8px 16px; background: #007bff; color: white; text-decoration: none; border-radius: 4px; font-size: 14px;">
                원문 보기 →
            </a>
        </div>"""
    
    def send_email_report(self, html_content: str, news_list: List[NewsItem]) -> bool:
        """이메일로 리포트 전송"""
        try:
            high_risk_count = len([n for n in news_list if n.risk_level == 'HIGH'])
            
            subject = f"[리스크 모니터링] {datetime.now().strftime('%Y-%m-%d')} - "
            if high_risk_count > 0:
                subject += f"⚠️ HIGH RISK {high_risk_count}건 발생"
            else:
                subject += "정상 모니터링 완료"
            
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
                
            logger.info(f"📧 이메일 전송 성공: {', '.join(self.email_config['recipients'])}")
            return True
            
        except Exception as e:
            logger.error(f"이메일 전송 실패: {e}")
            return False

    def run_daily_monitoring(self):
        """일일 전체 모니터링 (오전 7시 실행)"""
        logger.info("\n" + "="*70)
        logger.info("🌅 일일 전체 리스크 모니터링 시작")
        logger.info("="*70)
        
        try:
            # 전체 뉴스 수집 (국가별 + 회사)
            all_news = self.collect_all_news()
            
            # AI 분석 프로세스
            unique_news = self.analyzer.remove_duplicates(all_news)
            self.stats['news_after_dedup'] = len(unique_news)
            
            analyzed_news = self.analyzer.analyze_risk_batch(unique_news)
            self.stats['news_analyzed'] = len(analyzed_news)
            
            final_news = self.analyzer.summarize_and_translate(analyzed_news)
            
            # 통계 업데이트
            self.update_risk_statistics(final_news)
            
            # 리포트 생성 및 이메일 전송
            html_content = self.create_ai_html_report(final_news)
            
            # 파일 저장
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            html_file = f'daily_risk_report_{timestamp}.html'
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # 이메일 전송
            if self.email_config['sender_email'] and self.email_config['recipients']:
                self.send_email_report(html_content, final_news)

            # 상세 통계 로그 출력
            self._print_detailed_stats()

            logger.info("✅ 일일 모니터링 완료")
            return True
            
        except Exception as e:
            logger.error(f"일일 모니터링 오류: {e}")
            return False

    def _print_detailed_stats(self):
        """상세 통계 출력 - 새로운 메서드"""
        duration = datetime.now() - self.stats['start_time']
        
        logger.info("\n" + "="*70)
        logger.info("📊 상세 통계 보고서")
        logger.info("="*70)
        logger.info(f"실행 시간: {str(duration).split('.')[0]}")
        logger.info(f"API 호출 횟수: {self.stats['api_calls']}")
        logger.info(f"수집된 뉴스: {self.stats['news_collected']}건")
        logger.info(f"중복 제거 후: {self.stats['news_after_dedup']}건")
        logger.info(f"AI 분석 완료: {self.stats['news_analyzed']}건")
        logger.info(f"최종 필터링: {self.stats['total_filtered']}건")
        logger.info("-" * 70)
        logger.info("리스크 레벨별 분포:")
        logger.info(f"  HIGH: {self.stats['high_risk']}건")
        logger.info(f"  MEDIUM: {self.stats['medium_risk']}건")
        logger.info(f"  LOW: {self.stats['low_risk']}건")
        logger.info(f"  COMPANY: {self.stats['company_news']}건")
        logger.info("-" * 70)
        
        if self.stats['country_breakdown']:
            logger.info("국가별 리스크 분포 (상위 5개국):")
            sorted_countries = sorted(
                self.stats['country_breakdown'].items(), 
                key=lambda x: x[1]['total'], 
                reverse=True
            )[:5]
            
            for country, stats in sorted_countries:
                logger.info(f"  {country}: 총 {stats['total']}건 "
                          f"(H:{stats['high']}, M:{stats['medium']}, "
                          f"L:{stats['low']}, C:{stats['company']})")
        
        if self.stats['errors'] > 0:
            logger.warning(f"⚠️ 발생한 오류: {self.stats['errors']}건")
        
        logger.info("="*70)

    def run_company_monitoring(self, company_cache: CompanyNewsCache):
        """회사 관련 뉴스만 모니터링 (3시간 주기)"""
        logger.info("\n" + "="*70)
        logger.info("🏢 회사 관련 뉴스 모니터링 시작 (3시간 주기)")
        logger.info("="*70)
        
        try:
            # 회사 뉴스만 수집
            company_news = self.collect_company_news_only()
            
            # 새로운 뉴스만 필터링
            new_news = []
            for news in company_news:
                if company_cache.is_new_news(news.news_hash):
                    new_news.append(news)
                    company_cache.add_news(news.news_hash)
            
            if not new_news:
                logger.info("ℹ️ 새로운 회사 관련 뉴스 없음")
                return True
            
            logger.info(f"🆕 {len(new_news)}건의 새로운 회사 뉴스 발견")
            
            # AI 분석
            unique_news = self.analyzer.remove_duplicates(new_news)
            analyzed_news = self.analyzer.analyze_risk_batch(unique_news)
            final_news = self.analyzer.summarize_and_translate(analyzed_news)
            
            # 리스크 레벨별 분류
            high_risk = [n for n in final_news if n.risk_level == 'HIGH']
            medium_risk = [n for n in final_news if n.risk_level == 'MEDIUM']
            low_risk = [n for n in final_news if n.risk_level == 'LOW']
            company_level = [n for n in final_news if n.risk_level == 'COMPANY']
            
            # 1. 일반 수신자: HIGH 또는 MEDIUM이 있는 경우만 전송
            if (high_risk or medium_risk) and self.email_config['recipients']:
                # HIGH와 MEDIUM만 포함한 리포트 생성
                urgent_news = high_risk + medium_risk
                html_content = self.create_urgent_company_report(urgent_news, report_type='urgent')
                
                risk_text = []
                if high_risk:
                    risk_text.append(f"HIGH {len(high_risk)}건")
                if medium_risk:
                    risk_text.append(f"MEDIUM {len(medium_risk)}건")
                
                subject = f"[긴급] 삼성물산 관련 뉴스 - {' / '.join(risk_text)} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                
                # 일반 수신자에게 전송
                self.send_email_to_recipients(html_content, subject, self.email_config['recipients'])
                logger.info(f"📧 긴급 알림 이메일 전송 완료 (일반 수신자: {len(urgent_news)}건)")
            
            # 2. 관리자: 모든 뉴스 전송 (항상)
            if final_news and self.email_config.get('admin_email'):
                html_content_admin = self.create_urgent_company_report(final_news, report_type='admin')
                
                subject_admin = f"[관리자] 삼성물산 모니터링 - 전체 {len(final_news)}건 - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                
                # 관리자에게 전송
                self.send_email_to_recipients(html_content_admin, subject_admin, [self.email_config['admin_email']])
                logger.info(f"📧 관리자 전체 리포트 전송 완료 ({len(final_news)}건)")
            
            # 캐시 저장
            company_cache.save_cache()
            logger.info("✅ 회사 모니터링 완료")
            return True
            
        except Exception as e:
            logger.error(f"회사 모니터링 오류: {e}")
            return False

    def collect_company_news_only(self) -> List[NewsItem]:
        """회사 관련 뉴스만 수집 (3시간 주기용)"""
        all_news = []
        
        # 제외할 소스들
        korean_sources = ['yonhap', '연합', 'korea', 'chosun', '조선', 
                        'joongang', '중앙', 'hankyoreh', '한겨레', 'donga', '동아',
                        'hankook', '한국', 'maeil', '매일', 'seoul', '서울']
        
        official_sources = ['samsung newsroom', '삼성 뉴스룸', 'samsung.com', 
                        'samsungcnt.com', 'samsung c&t newsroom']
        
        construction_keywords = ['construction', 'building', 'infrastructure', 'engineering',
                            'project', 'development', 'contractor', 'architecture',
                            '건설', '건축', '공사', '시공', '프로젝트', '개발']
        
        logger.info("\n🏢 회사 키워드 뉴스 수집 시작 (해외만)")
        
        for idx, keyword in enumerate(self.company_keywords, 1):
            logger.info(f"[{idx}/{len(self.company_keywords)}] {keyword}")
            
            # 건설업 관련 검색어
            query = f'"{keyword}" (construction OR building OR project OR infrastructure) -site:kr -korea -한국 -newsroom'
            
            try:
                params = {
                    "api_key": self.serpapi_key,
                    "engine": "google_news",
                    "q": query,
                    "when": "7d",
                    "gl": "us",
                    "hl": "en"
                }
                
                search = self.GoogleSearch(params)
                response = search.get_dict()
                
                if "news_results" in response:
                    for item in response["news_results"][:30]:
                        # 날짜 체크
                        if not self._is_within_days(item.get('date', ''), 7):
                            continue
                        
                        source = item.get('source', {}).get('name', '').lower()
                        
                        # 필터링
                        if any(ks in source for ks in korean_sources):
                            continue
                        
                        if any(os in source for os in official_sources):
                            continue
                        
                        # 건설업 관련성 체크
                        title = item.get('title', '').lower()
                        snippet = item.get('snippet', '').lower()
                        
                        has_construction_relevance = any(
                            ck.lower() in title or ck.lower() in snippet 
                            for ck in construction_keywords
                        )
                        
                        if not has_construction_relevance:
                            continue
                        
                        news_item = NewsItem(
                            title=item.get('title', ''),
                            date=item.get('date', ''),
                            source=item.get('source', {}).get('name', 'Unknown'),
                            snippet=item.get('snippet', ''),
                            link=item.get('link', ''),
                            country="Global",
                            country_ko="해외",
                            country_code="global_samsung",
                            search_type='company_global',
                            collected_at=datetime.now().isoformat()
                        )
                        all_news.append(news_item)
                        
                        if len(all_news) >= 10:  # 최대 10건
                            break
                    
            except Exception as e:
                logger.error(f"회사 키워드 검색 오류: {e}")
            
            time.sleep(1)
        
        # 2. 한국 미디어에서 회사 검색
        logger.info("\n🇰🇷 한국 언론 내 회사 뉴스 모니터링")
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
                
                # 회사 관련 뉴스 표시
                for item in news:
                    item.country = "Samsung C&T"
                    item.country_ko = "삼성물산"
                    item.country_code = "samsung"
                
                all_news.extend(news)
                time.sleep(1)
        
        logger.info(f"\n✅ 회사 관련 뉴스 {len(all_news)}건 수집 완료")
        return all_news

    def create_urgent_company_report(self, news_list: List[NewsItem], report_type: str = 'urgent') -> str:
        """긴급 회사 뉴스 이메일 리포트 생성
        
        Args:
            news_list: 뉴스 리스트
            report_type: 'urgent' (HIGH/MEDIUM만) 또는 'admin' (전체)
        """
        # 리스크 레벨별 분류
        high_risk = [n for n in news_list if n.risk_level == 'HIGH']
        medium_risk = [n for n in news_list if n.risk_level == 'MEDIUM']
        low_risk = [n for n in news_list if n.risk_level == 'LOW']
        company_news = [n for n in news_list if n.risk_level == 'COMPANY']
        
        # 전체 뉴스 개수
        total_news = len(news_list)
        
        # 리포트 타입에 따른 제목과 색상
        if report_type == 'admin':
            header_color = '#17a2b8'  # 청록색 (관리자용)
            header_title = "📊 삼성물산 전체 모니터링 리포트 (관리자)"
            alert_message = f"전체 {total_news}건의 뉴스가 감지되었습니다."
        else:
            header_color = '#dc3545' if high_risk else '#ffc107'
            header_title = "⚠️ 삼성물산 관련 긴급 뉴스"
            alert_message = f"중요도 높은 뉴스 {len(high_risk) + len(medium_risk)}건이 감지되었습니다."
        
        html = f"""<!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>삼성물산 관련 뉴스</title>
    </head>
    <body style="margin: 0; padding: 0; font-family: 'Malgun Gothic', Arial, sans-serif; background-color: #f4f4f4;">
        <div style="max-width: 700px; margin: 0 auto; background-color: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <div style="background-color: {header_color}; padding: 25px; text-align: center;">
                <h1 style="color: #ffffff; margin: 0; font-size: 26px;">{header_title}</h1>
                <p style="color: #ffffff; margin: 10px 0 0 0; font-size: 14px;">
                    {datetime.now().strftime('%Y년 %m월 %d일 %H:%M')} | {total_news}건 감지
                </p>
            </div>
            
            <div style="padding: 25px;">
                <div style="background-color: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin-bottom: 20px;">
                    <p style="margin: 0; color: #856404;">
                        <strong>알림:</strong> {alert_message}
                    </p>
                    <p style="margin: 5px 0 0 0; color: #856404; font-size: 13px;">
                        HIGH: {len(high_risk)}건 | MEDIUM: {len(medium_risk)}건 | LOW: {len(low_risk)}건 | COMPANY: {len(company_news)}건
                    </p>
                </div>"""
        
        news_counter = 1
        
        # HIGH RISK 뉴스
        if high_risk:
            html += f"""
                <h2 style="color: #dc3545; margin: 25px 0 15px 0; font-size: 20px; border-bottom: 2px solid #dc3545; padding-bottom: 5px;">
                    🔴 HIGH RISK ({len(high_risk)})
                </h2>"""
            
            for news in high_risk:
                html += self._create_urgent_news_item(news, news_counter, '#dc3545')
                news_counter += 1
        
        # MEDIUM RISK 뉴스
        if medium_risk:
            html += f"""
                <h2 style="color: #ffc107; margin: 25px 0 15px 0; font-size: 20px; border-bottom: 2px solid #ffc107; padding-bottom: 5px;">
                    🟡 MEDIUM RISK ({len(medium_risk)})
                </h2>"""
            
            for news in medium_risk:
                html += self._create_urgent_news_item(news, news_counter, '#ffc107')
                news_counter += 1
        
        # 관리자 리포트인 경우에만 LOW와 COMPANY 포함
        if report_type == 'admin':
            # LOW RISK 뉴스
            if low_risk:
                html += f"""
                    <h2 style="color: #28a745; margin: 25px 0 15px 0; font-size: 20px; border-bottom: 2px solid #28a745; padding-bottom: 5px;">
                        🟢 LOW RISK ({len(low_risk)})
                    </h2>"""
                
                for news in low_risk:
                    html += self._create_urgent_news_item(news, news_counter, '#28a745')
                    news_counter += 1
            
            # COMPANY 뉴스
            if company_news:
                html += f"""
                    <h2 style="color: #6c757d; margin: 25px 0 15px 0; font-size: 20px; border-bottom: 2px solid #6c757d; padding-bottom: 5px;">
                        🏢 COMPANY NEWS ({len(company_news)})
                    </h2>"""
                
                for news in company_news:
                    html += self._create_urgent_news_item(news, news_counter, '#6c757d')
                    news_counter += 1
        
        html += """
            </div>
            <div style="background-color: #f8f9fa; padding: 20px; text-align: center; border-top: 1px solid #dee2e6;">
                <p style="margin: 0; color: #666; font-size: 12px;">
                    Samsung C&T Risk Monitoring System<br>
                    3시간 주기 자동 모니터링
                </p>
            </div>
        </div>
    </body>
    </html>"""
        return html

    def _create_urgent_news_item(self, news: NewsItem, idx: int, border_color: str) -> str:
        """개별 뉴스 아이템 HTML 생성 (헬퍼 메소드)"""
        title_to_display = news.ai_title_ko if news.ai_title_ko else news.title
        
        # 리스크 정보 표시 (COMPANY는 카테고리만)
        if news.risk_level == 'COMPANY':
            risk_info = f"카테고리: {news.risk_category or 'Company News'}"
        else:
            risk_info = f"<strong style='color: {border_color};'>리스크 점수: {news.risk_score:.0f}</strong> | 카테고리: {news.risk_category or 'Other'}"
        
        return f"""
        <div style="border: 1px solid #dee2e6; border-left: 5px solid {border_color}; padding: 20px; margin-bottom: 20px; background-color: #f8f9fa;">
            <h3 style="margin: 0 0 10px 0; color: #333; font-size: 18px;">
                {idx}. {title_to_display}
            </h3>
            <div style="margin: 10px 0; color: #666; font-size: 13px;">
                📰 {news.source} | 📅 {news.date}
            </div>
            <div style="margin: 15px 0; padding: 10px; background-color: #ffffff; border-radius: 4px;">
                {risk_info}
            </div>
            <div style="margin: 15px 0; padding: 10px; background-color: #ffffff; border-radius: 4px;">
                <strong>AI 요약:</strong><br>
                <p style="margin: 5px 0; color: #333; line-height: 1.6;">
                    {news.ai_summary_ko or '요약 생성 중...'}
                </p>
            </div>
            <a href="{news.link}" style="display: inline-block; margin-top: 10px; padding: 10px 20px; background-color: #007bff; color: white; text-decoration: none; border-radius: 4px;">
                원문 보기 →
            </a>
        </div>"""

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

    def send_email_to_recipients(self, html_content: str, subject: str, recipients: List[str]) -> bool:
        """특정 수신자들에게 이메일 전송"""
        try:
            msg = MIMEMultipart('alternative')
            msg['Subject'] = subject
            msg['From'] = self.email_config['sender_email']
            msg['To'] = ', '.join(recipients)
            
            html_part = MIMEText(html_content, 'html', 'utf-8')
            msg.attach(html_part)
            
            with smtplib.SMTP(self.email_config['smtp_server'], self.email_config['smtp_port']) as server:
                server.starttls()
                server.login(self.email_config['sender_email'], self.email_config['sender_password'])
                server.send_message(msg)
                
            logger.info(f"📧 이메일 전송 성공: {', '.join(recipients)}")
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
            self.update_risk_statistics(final_news)
            
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
            self._print_detailed_stats()
            
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


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description='AI 기반 글로벌 리스크 모니터링')
    parser.add_argument('--mode', 
                       choices=['test', 'daily', 'company', 'schedule'],
                       default='test',
                       help='실행 모드: test(테스트-1회), daily(일일전체), company(회사만), schedule(스케줄링)')
    parser.add_argument('--config', 
                       default='monitoring_config.json', 
                       help='설정 파일 경로')
    args = parser.parse_args()
    
    try:
        monitor = AIRiskMonitoringSystem(args.config)
        
        if args.mode == 'test':
            logger.info("\n🧪 테스트 모드 - 전체 모니터링 1회 실행")
            result = monitor.run()
            
            # 테스트 모드에서도 명시적으로 이메일 전송 확인
            if result['success']:
                logger.info("✅ 테스트 모드 실행 완료")
                if monitor.email_config['sender_email'] and monitor.email_config['recipients']:
                    logger.info("📧 이메일이 설정되어 있으면 자동 전송됩니다.")
                else:
                    logger.warning("⚠️ 이메일 설정이 없어 전송되지 않았습니다.")
            
        elif args.mode == 'daily':
            logger.info("\n📊 일일 전체 모니터링 모드")
            monitor.run_daily_monitoring()
            
        elif args.mode == 'company':
            logger.info("\n🏢 회사 전용 모니터링 모드")
            company_cache = CompanyNewsCache()
            monitor.run_company_monitoring(company_cache)
            
        elif args.mode == 'schedule':
            logger.info("\n⏰ 스케줄 모드 시작")
            logger.info("설정된 스케줄:")
            logger.info("  - 매일 오전 7시: 전체 리스크 모니터링")
            logger.info("  - 3시간마다: 회사 관련 뉴스 모니터링")
            logger.info("Ctrl+C로 중단할 수 있습니다.\n")
            
            company_cache = CompanyNewsCache()
            
            # 스케줄 설정
            schedule.every().day.at("07:00").do(monitor.run_daily_monitoring)
            schedule.every(3).hours.do(monitor.run_company_monitoring, company_cache)
            
            # 시작 시 즉시 한 번 실행 (회사 모니터링만)
            logger.info("🚀 초기 실행 시작...")
            monitor.run_company_monitoring(company_cache)
            
            # 스케줄 루프 실행
            logger.info("\n⏳ 스케줄러 대기 중...")
            while True:
                schedule.run_pending()
                time.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("\n사용자에 의해 중단되었습니다")
    except Exception as e:
        logger.error(f"치명적 오류: {e}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main()