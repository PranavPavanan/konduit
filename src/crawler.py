"""
Web crawler module with robots.txt compliance and polite crawling
"""

import asyncio
import aiohttp
import time
import logging
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser
from bs4 import BeautifulSoup
from readability import Document
import re
from typing import Set, List, Dict, Optional, Tuple
from dataclasses import dataclass
import random

logger = logging.getLogger(__name__)

@dataclass
class CrawledPage:
    """Data class for crawled page content"""
    url: str
    title: str
    content: str
    links: List[str]
    depth: int
    timestamp: float

class CrawlerService:
    """Web crawler service with robots.txt compliance and polite crawling"""
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.visited_urls: Set[str] = set()
        self.robots_cache: Dict[str, RobotFileParser] = {}
        self.crawled_pages: List[CrawledPage] = []
        self.domain_limits: Dict[str, float] = {}  # Track last request time per domain
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self._init_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self._close_session()
    
    async def _init_session(self):
        """Initialize aiohttp session with proper headers"""
        timeout = aiohttp.ClientTimeout(total=30, connect=10)
        headers = {
            'User-Agent': 'RAG-Service-Crawler/1.0 (+https://example.com/bot)',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        }
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers=headers,
            connector=aiohttp.TCPConnector(limit=10, limit_per_host=2)
        )
    
    async def _close_session(self):
        """Close aiohttp session"""
        if self.session:
            await self.session.close()
    
    def _normalize_url(self, url: str, base_url: str) -> Optional[str]:
        """Normalize URL and check if it's within the same domain"""
        try:
            parsed_url = urlparse(url)
            parsed_base = urlparse(base_url)
            
            # Handle relative URLs
            if not parsed_url.netloc:
                url = urljoin(base_url, url)
                parsed_url = urlparse(url)
            
            # Only crawl same domain
            if parsed_url.netloc != parsed_base.netloc:
                return None
                
            # Normalize URL (remove fragments, sort query params)
            normalized = urlunparse((
                parsed_url.scheme,
                parsed_url.netloc,
                parsed_url.path,
                parsed_url.params,
                parsed_url.query,
                ''  # Remove fragment
            ))
            
            # Only crawl HTML pages
            if not any(normalized.lower().endswith(ext) for ext in ['.html', '.htm', '']):
                return None
                
            return normalized
        except Exception as e:
            logger.warning(f"URL normalization failed for {url}: {e}")
            return None
    
    async def _check_robots_txt(self, url: str) -> bool:
        """Check if URL is allowed by robots.txt"""
        try:
            parsed_url = urlparse(url)
            robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
            
            if robots_url not in self.robots_cache:
                rp = RobotFileParser()
                rp.set_url(robots_url)
                rp.read()
                self.robots_cache[robots_url] = rp
            
            rp = self.robots_cache[robots_url]
            return rp.can_fetch('*', url)
        except Exception as e:
            logger.warning(f"Robots.txt check failed for {url}: {e}")
            return True  # Allow by default if robots.txt is inaccessible
    
    async def _respect_crawl_delay(self, url: str, delay_ms: int):
        """Respect crawl delay and rate limiting"""
        domain = urlparse(url).netloc
        current_time = time.time()
        
        if domain in self.domain_limits:
            time_since_last = current_time - self.domain_limits[domain]
            required_delay = delay_ms / 1000.0
            
            if time_since_last < required_delay:
                sleep_time = required_delay - time_since_last
                # Add small random delay to avoid thundering herd
                sleep_time += random.uniform(0.1, 0.5)
                logger.debug(f"Respecting crawl delay: sleeping {sleep_time:.2f}s for {domain}")
                await asyncio.sleep(sleep_time)
        
        self.domain_limits[domain] = time.time()
    
    async def _extract_content(self, html: str, url: str) -> Tuple[str, str, List[str]]:
        """Extract main content, title, and links from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Extract title
            title = ""
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text().strip()
            
            # Use readability to extract main content
            doc = Document(html)
            content_html = doc.summary()
            
            # Parse the cleaned HTML to extract text
            content_soup = BeautifulSoup(content_html, 'html.parser')
            
            # Remove script and style elements
            for script in content_soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # Extract text content
            content = content_soup.get_text()
            
            # Clean up text
            content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
            content = content.strip()
            
            # Extract links
            links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                normalized_url = self._normalize_url(href, url)
                if normalized_url and normalized_url not in self.visited_urls:
                    links.append(normalized_url)
            
            return title, content, links
            
        except Exception as e:
            logger.error(f"Content extraction failed for {url}: {e}")
            return "", "", []
    
    async def _crawl_page(self, url: str, depth: int, max_depth: int, 
                         delay_ms: int) -> Optional[CrawledPage]:
        """Crawl a single page"""
        try:
            # Check robots.txt
            if not await self._check_robots_txt(url):
                logger.info(f"Robots.txt disallows crawling {url}")
                return None
            
            # Respect crawl delay
            await self._respect_crawl_delay(url, delay_ms)
            
            # Fetch page
            async with self.session.get(url) as response:
                if response.status != 200:
                    logger.warning(f"HTTP {response.status} for {url}")
                    return None
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if 'text/html' not in content_type:
                    logger.info(f"Skipping non-HTML content: {url}")
                    return None
                
                html = await response.text()
                
                # Extract content
                title, content, links = await self._extract_content(html, url)
                
                if not content or len(content) < 100:  # Skip pages with too little content
                    logger.info(f"Skipping page with insufficient content: {url}")
                    return None
                
                page = CrawledPage(
                    url=url,
                    title=title,
                    content=content,
                    links=links,
                    depth=depth,
                    timestamp=time.time()
                )
                
                logger.info(f"Crawled page: {url} (depth {depth}, {len(content)} chars)")
                return page
                
        except Exception as e:
            logger.error(f"Failed to crawl {url}: {e}")
            return None
    
    async def crawl(self, start_url: str, max_pages: int = 50, 
                   max_depth: int = 3, crawl_delay_ms: int = 1000) -> Dict:
        """Main crawling method using BFS"""
        start_time = time.time()
        
        # Initialize session
        await self._init_session()
        
        try:
            # Normalize start URL
            normalized_start = self._normalize_url(start_url, start_url)
            if not normalized_start:
                raise ValueError(f"Invalid start URL: {start_url}")
            
            # BFS queue: (url, depth)
            queue = [(normalized_start, 0)]
            self.visited_urls.add(normalized_start)
            
            page_count = 0
            skipped_count = 0
            errors = []
            
            while queue and page_count < max_pages:
                url, depth = queue.pop(0)
                
                # Skip if depth exceeded
                if depth > max_depth:
                    skipped_count += 1
                    continue
                
                # Crawl page
                page = await self._crawl_page(url, depth, max_depth, crawl_delay_ms)
                
                if page:
                    self.crawled_pages.append(page)
                    page_count += 1
                    
                    # Add new links to queue (only if we haven't reached max pages)
                    if page_count < max_pages and depth < max_depth:
                        for link in page.links:
                            if link not in self.visited_urls:
                                self.visited_urls.add(link)
                                queue.append((link, depth + 1))
                else:
                    skipped_count += 1
            
            crawl_time = time.time() - start_time
            
            return {
                "page_count": page_count,
                "skipped_count": skipped_count,
                "urls": [page.url for page in self.crawled_pages],
                "errors": errors,
                "crawl_time_seconds": crawl_time
            }
            
        finally:
            await self._close_session()
    
    def get_crawled_pages(self) -> List[CrawledPage]:
        """Get all crawled pages"""
        return self.crawled_pages.copy()
    
    def clear_cache(self):
        """Clear crawled pages and visited URLs"""
        self.crawled_pages.clear()
        self.visited_urls.clear()
        self.robots_cache.clear()
        self.domain_limits.clear()
